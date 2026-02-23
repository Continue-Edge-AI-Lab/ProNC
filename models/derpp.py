# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.buffer import Buffer, fill_tsne_buffer
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


class Derpp(ContinualModel):
    """Continual learning via Dark Experience Replay++."""
    NAME = 'derpp'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    @staticmethod
    def get_parser(parser) -> ArgumentParser:
        add_rehearsal_args(parser)
        parser.add_argument('--alpha', type=float, required=True,
                            help='Penalty weight.')
        parser.add_argument('--beta', type=float, required=True,
                            help='Penalty weight.')
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        self.buffer = Buffer(self.args.buffer_size)

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):

        self.opt.zero_grad()

        outputs = self.net(inputs)

        loss = self.loss(outputs, labels)
        loss.backward()
        tot_loss = loss.item()

        if not self.buffer.is_empty():
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_mse = self.args.alpha * F.mse_loss(buf_outputs, buf_logits)
            loss_mse.backward()
            tot_loss += loss_mse.item()

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=self.transform, device=self.device)

            buf_outputs = self.net(buf_inputs)
            loss_ce = self.args.beta * self.loss(buf_outputs, buf_labels)
            loss_ce.backward()
            tot_loss += loss_ce.item()

        self.opt.step()

        self.buffer.add_data(examples=not_aug_inputs,
                             labels=labels,
                             logits=outputs.data)

        return tot_loss
    
    def begin_task(self, dataset) -> None:
        self.tsne_buffer = Buffer(self.dataset.N_CLASSES_PER_TASK * 20)
    
    def end_task(self, dataset) -> None:
        with torch.no_grad():
            fill_tsne_buffer(self.tsne_buffer, dataset, self.current_task, net=self.net, use_herding=True)
        
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        folder_path = f"3d/tsne/derpp/{self.dataset.NAME}_20"
        file_name = f"tsne_plot_task{self.current_task}.png"
        directory = os.path.join(parent_dir, folder_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        file_path = os.path.join(directory, file_name)
        
        with torch.no_grad():
            tsne_inputs, tsen_labels = self.tsne_buffer.get_all_data()
            tsne_features = self.net(tsne_inputs.to(self.device), returnt='features')
        tsne_features = tsne_features.detach().cpu().numpy()
        tsne_labels_np = tsen_labels.detach().cpu().numpy()

        #pca = PCA(n_components=30)
        #tsne_features = pca.fit_transform(tsne_features)
        tsne = TSNE(n_components=3, perplexity=20, learning_rate='auto', random_state=42, early_exaggeration=20)
        features_tsne = tsne.fit_transform(tsne_features)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(features_tsne[:, 0], features_tsne[:, 1], features_tsne[:, 2], c=tsne_labels_np, cmap='tab10', alpha=0.7)
        plt.legend(*scatter.legend_elements(), title="Classes")
        plt.title("t-SNE Visualization of Last-Layer Features")
        plt.savefig(file_path)
        
