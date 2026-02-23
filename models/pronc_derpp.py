# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import get_dataset
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, fill_buffer, icarl_replay, fill_tsne_buffer
from utils.drloss import DRLoss
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN




class extend_orth(ContinualModel):
    """Continual Learning via extend orthignal matrix of neural collaps."""
    NAME = 'pronc_derpp'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--main_weight', type=float, required=False, default=1.,
                            help='Weight for main dr loss')
        parser.add_argument('--ce_weight', type=float, required=False, default=1.,
                            help='Weight for ce loss')
        parser.add_argument('--distill_weight', type=float, required=False, default=1.,
                            help='Weight for distill loss')
        parser.add_argument('--base_task', type=int, required=False, default=1,
                            help='Number of base task')
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.dataset = get_dataset(args)

        self.buffer = Buffer(self.args.buffer_size)
        self.orth = None
        self.etf = None
        self.old_net = None
        self.loss = DRLoss()
        self.ce_loss = F.cross_entropy
        self.main_weight = args.main_weight
        self.distill_weight = args.distill_weight
        self.ce_weight = args.ce_weight
        self.base_task = args.base_task
        self.base_num = args.base_task * self.dataset.N_CLASSES_PER_TASK
        self.loss_df = {}
        self.dr_loss = None
        self.dist_loss = None
        self.ce_loss_item = None
        self.total_loss = None
        self.mse_loss = None
        self.old_etf = []
        self.class_mean = None
        self.all_old_class_mean = []
        self.class_to_idx_map = None
        self.class_num = 0
        self.noval_data_loader = None
        self.noval_class_mean = None
        self.tsne_inputs = []
        self.tsne_labels = []
        
    def forward(self, x):
        with torch.no_grad():
            if (self.current_task > 1) & (self.args.main_weight>0):
                feature = self.net(x, returnt='features')
                test_proto = self.etf[:, 0:self.class_num]
                logits = feature @ test_proto
            else:
                logits = self.net(x)
        return logits

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.opt.zero_grad()

        if self.current_task == 0:
            outputs = self.net(inputs)

            loss = self.ce_loss(outputs, labels)

            total_loss = loss.item()

        else:
            self.class_num = self.dataset.N_CLASSES_PER_TASK * (self.current_task+1)
            etf_vec = deepcopy(self.etf.T)
            use_proto = etf_vec

            # new task
            outputs, features = self.net(inputs, returnt='both')

            with torch.no_grad():
                old_features = self.old_net(inputs, returnt='features')

            # dr loss and distill for ours
            loss = self.main_weight * self.loss(F.normalize(features, dim=1), use_proto[labels, :].to(self.device))
            total_loss = loss.item()
            distill_loss = self.distill_weight * self.loss(F.normalize(features, dim=1), F.normalize(old_features, dim=1))
            loss += distill_loss

            total_loss += distill_loss.item()

            # ce loss for DER++
            ce_loss= self.ce_weight * self.ce_loss(outputs, labels)
            loss += ce_loss
            total_loss += ce_loss.item()
            # buffer distill
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)
            with torch.no_grad():
                old_buf_logits, old_buf_features = self.old_net(buf_inputs, returnt='both')

            buf_outputs, buf_features = self.net(buf_inputs, returnt='both')

            #distill for DER++
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, old_buf_logits)
            loss += loss_mse
            total_loss += loss_mse.item()

            # distill for ours
            distill_loss = self.distill_weight * self.loss(F.normalize(buf_features, dim=1), F.normalize(old_buf_features, dim=1))
            loss += distill_loss
            total_loss += distill_loss.item()

            # buffer replay for DER++
            loss_ce = self.args.beta * self.ce_loss(buf_outputs, buf_labels)
            loss += loss_ce
            total_loss += loss_ce.item()

            # buffer replay for ours
            loss_dr = self.main_weight * self.loss(F.normalize(buf_features, dim=1), use_proto[buf_labels, :].to(self.device))
            loss += loss_dr
            total_loss += loss_dr.item()

        loss.backward()

        self.opt.step()

        return total_loss

    def begin_task(self, dataset):

        self.noval_data_loader = dataset.train_loader
        if self.current_task == 1:
            if self.base_task == 1:
                self.gram_schmidt_extend(self.dataset.N_CLASSES_PER_TASK)
            else:
                self.gram_schmidt_extend(self.base_num - self.dataset.N_CLASSES_PER_TASK)
            self._init_etf()
        elif self.current_task >= self.base_task:
            self.gram_schmidt_extend(self.dataset.N_CLASSES_PER_TASK)
            self._init_etf()


    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval()).to(self.device)
        if self.current_task == 0:
            self.noval_class_mean = self.get_noval_classmen()
        self.net.train()

        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=False)

        if self.current_task == 0:
            self.get_simi_orth()
            self.closest_orth_matrix()
            self._init_etf()



    @torch.no_grad()
    def get_class_mean(self, inputs, labels) -> None:
        self.net.eval()
        old_class = (self.current_task+1) * self.dataset.N_CLASSES_PER_TASK
        class_feature_sums = torch.zeros(old_class, 512).to(self.device)
        class_counts = torch.zeros(old_class).to(self.device)

        with torch.no_grad():
            features = self.net(inputs.to(self.device), returnt='features')
            for i in range(labels.size(0)):
                class_feature_sums[labels[i]] += features[i]
                class_counts[labels[i]] += 1

        class_mean = class_feature_sums / class_counts.view(-1, 1)
        self.class_mean = class_mean
        
    def get_noval_classmen(self):
        data_loader = self.noval_data_loader
        class_feature_sums = torch.zeros(self.dataset.N_CLASSES_PER_TASK, 512).to(self.device)
        class_counts = torch.zeros(self.dataset.N_CLASSES_PER_TASK).to(self.device)
        train_iter = iter(data_loader)
        while True:
            try:
                data = next(train_iter)
            except StopIteration:
                break
            self.net.eval()
            inputs, labels, _ = data[0], data[1], data[2]
            inputs, labels = inputs.to(self.device), labels.to(self.device, dtype=torch.long)
            with torch.no_grad():
                features = F.normalize(self.net(inputs, returnt='features'), dim=1)
                for i in range(labels.size(0)):
                    if labels[i] - self.current_task * self.dataset.N_CLASSES_PER_TASK >= 0 :
                        class_feature_sums[labels[i] - self.current_task * self.dataset.N_CLASSES_PER_TASK] += features[i]
                        class_counts[labels[i] - self.current_task * self.dataset.N_CLASSES_PER_TASK] += 1

        noval_class_mean = class_feature_sums / class_counts.view(-1, 1)
        return noval_class_mean


    def get_simi_orth(self) -> None:
        scaling_factor = math.sqrt(self.base_num / (self.base_num - 1))
        norm_center_class_mean = F.normalize(self.noval_class_mean - torch.mean(self.noval_class_mean, dim = 0), dim=1)
        intermediate = norm_center_class_mean.T / scaling_factor

        i_nc_nc = torch.eye(self.dataset.N_CLASSES_PER_TASK)  
        one_nc_nc = torch.ones(self.dataset.N_CLASSES_PER_TASK, self.dataset.N_CLASSES_PER_TASK) / self.dataset.N_CLASSES_PER_TASK  # Matrix of ones scaled by 1/ac
        M = i_nc_nc - one_nc_nc  

        M_inv = torch.linalg.pinv(M).to(self.device) 
        orth_recovered = torch.matmul(intermediate, M_inv)
        self.orth = orth_recovered

    def closest_orth_matrix(self) -> None:
        U, _, Vt = torch.linalg.svd(self.orth, full_matrices=False)
        Q = torch.matmul(U, Vt).to('cpu')
        self.orth = Q.numpy()


    def _init_etf(self) -> None:
        ac = self.orth.shape[1]
        i_nc_nc = torch.eye(ac)
        one_nc_nc = torch.ones(ac, ac) / ac
        etf_vec = torch.mul(
            torch.matmul(torch.tensor(self.orth).float(), i_nc_nc - one_nc_nc),
            math.sqrt(ac / (ac - 1))
        )

        self.etf = F.normalize(etf_vec, dim=0).to(self.device)


    def gram_schmidt_extend(self, extend_num) -> None:
        rows, cols = self.orth.shape

        new_vectors = np.random.randn(rows, extend_num)

        for i in range(extend_num):

            v = new_vectors[:, i]

                v -= np.dot(self.orth[:, j], v) * self.orth[:, j]
            for j in range(i): 
                v -= np.dot(new_vectors[:, j], v) * new_vectors[:, j]

            v /= np.linalg.norm(v)

            new_vectors[:, i] = v

        orth_extended = np.hstack([self.orth, new_vectors])

        self.orth = orth_extended
