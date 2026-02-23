
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
from utils.buffer import Buffer, fill_buffer, icarl_replay
from utils.drloss import DRLoss
import numpy as np
import math
import pandas as pd



class extend_orth(ContinualModel):
    """Continual Learning via Neural Collapse Terminus."""
    NAME = 'exorth_NCT'
    COMPATIBILITY = ['class-il', 'task-il']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)

        parser.add_argument('--main_weight', type=float, required=False, default=1.,
                            help='Weight for main dr loss')
        parser.add_argument('--distill_weight', type=float, required=False, default=1.,
                            help='Weight for distill loss')

        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)
        self.dataset = get_dataset(args)

        # Instantiate buffers
        self.buffer = Buffer(self.args.buffer_size)
        self.orth = None
        self.etf = None
        self.old_net = None
        self.loss = DRLoss()
        self.main_weight = args.main_weight
        self.distill_weight = args.distill_weight
        self.loss_df = {}
        self.dr_loss = None
        self.dist_loss = None
        self.total_loss = None
        self.class_mean = None
        self.all_old_class_mean = []
        self.class_to_idx_map = None
        self.class_num = 0
        self.noval_data_loader = None
        self.noval_class_mean = None

    def forward(self, x):
        with torch.no_grad():
            feature = self.net(x, returnt='features')
            feature = F.normalize(feature, dim=1)
            test_proto = self.etf[:, 0:self.class_num]

            logits = feature @ test_proto


        return logits

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_num = self.dataset.N_CLASSES_PER_TASK * (self.current_task+1)
        start = self.dataset.N_CLASSES_PER_TASK * (self.current_task)

        self.opt.zero_grad()

        etf_vec = deepcopy(self.etf.T)

        if self.current_task > 0:
            etf_vec[start:self.class_num] = (epoch/self.args.n_epochs)*etf_vec[start:self.class_num] + (1-epoch/self.args.n_epochs)*self.noval_class_mean
            use_proto = etf_vec / torch.norm(etf_vec, p=2, dim=1, keepdim=True)
        else:
            use_proto = etf_vec

        outputs = self.net(inputs, returnt='features')

        loss = self.main_weight * self.loss(F.normalize(outputs, dim=1), use_proto[labels, :].to(self.device))
        


        if self.current_task > 0:
            old_feature = self.old_net(inputs, returnt='features')
            distill_loss = self.distill_weight * self.loss(F.normalize(outputs, dim=1), F.normalize(old_feature, dim=1))
            loss = loss + distill_loss
            
        loss.backward()
        self.opt.step()


        return loss

    def begin_epoch(self, epoch: int, dataset: ContinualDataset) -> None:
        if self.current_task > 0:
            self.noval_class_mean = self.get_noval_classmen()

    def begin_task(self, dataset):
        if self.current_task == 0:
            self._generate_random_orthogonal_matrix()
            self._init_etf()

        icarl_replay(self, dataset)
        



    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval()).to(self.device)
        self.net.train()
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=False)




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
        return F.normalize(noval_class_mean, dim=1)
        
        



    def _generate_random_orthogonal_matrix(self) -> None:
        rand_mat = np.random.random(size=(512, self.dataset.N_CLASSES))
        orth_vec, _ = np.linalg.qr(rand_mat)
        self.orth = orth_vec

    def _init_etf(self) -> None:
        ac = self.orth.shape[1]
        i_nc_nc = torch.eye(ac)
        one_nc_nc = torch.ones(ac, ac) / ac
        etf_vec = torch.mul(
            torch.matmul(torch.tensor(self.orth).float(), i_nc_nc - one_nc_nc),
            math.sqrt(ac / (ac - 1))
        )

        self.etf = etf_vec.to(self.device)
