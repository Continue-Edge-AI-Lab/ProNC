# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

import torch
import torch.nn.functional as F
from datasets import get_dataset
import math
import numpy as np
from models.utils.continual_model import ContinualModel
from utils.args import add_rehearsal_args, ArgumentParser
from utils.batch_norm import bn_track_stats
from utils.buffer import Buffer, fill_buffer, icarl_replay
from utils.drloss import DRLoss

class ICarl(ContinualModel):
    """Continual Learning via iCaRL."""
    NAME = 'icarl_ours'
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
        self.eye = torch.eye(self.num_classes).to(self.device)

        self.class_means = None
        self.old_net = None
        self.noval_data_loader = None
        self.dr_loss = DRLoss()
        self.orth = None
        self.etf = None
        self.old_net = None
        self.class_num = None

    def forward(self, x):
        if self.class_means is None:
            with torch.no_grad():
                self.compute_class_means()
                self.class_means = self.class_means.squeeze()

        feats = self.net(x, returnt='features')
        feats = feats.view(feats.size(0), -1)
        feats = feats.unsqueeze(1)

        pred = (self.class_means.unsqueeze(0) - feats).pow(2).sum(2)
        return -pred

    def observe(self, inputs, labels, not_aug_inputs, logits=None, epoch=None):

        if not hasattr(self, 'classes_so_far'):
            self.register_buffer('classes_so_far', labels.unique().to('cpu'))
        else:
            self.register_buffer('classes_so_far', torch.cat((
                self.classes_so_far, labels.to('cpu'))).unique())

        self.class_means = None
        if self.current_task > 0:

            with torch.no_grad():
                logits = torch.sigmoid(self.old_net(inputs))
        self.opt.zero_grad()
        loss = self.get_loss(inputs, labels, self.current_task, logits)

        if self.current_task > 0:

            self.class_num = self.dataset.N_CLASSES_PER_TASK * (self.current_task+1)
            features = self.net(inputs, returnt='features')
            etf_vec = deepcopy(self.etf.T)
            use_proto = etf_vec
            loss += self.args.main_weight * self.loss(F.normalize(features, dim=1), use_proto[labels, :].to(self.device))
            old_feature = self.old_net(inputs, returnt='features')
            distill_loss = self.args.distill_weight * self.loss(F.normalize(features, dim=1), F.normalize(old_feature, dim=1))
            loss += loss + distill_loss
        
        loss.backward()

        self.opt.step()

        return loss.item()

    @staticmethod
    def binary_cross_entropy(pred, y):
        return -(pred.log() * y + (1 - y) * (1 - pred).log()).mean()

    def get_loss(self, inputs: torch.Tensor, labels: torch.Tensor,
                 task_idx: int, logits: torch.Tensor) -> torch.Tensor:
        """
        Computes the loss tensor.

        Args:
            inputs: the images to be fed to the network
            labels: the ground-truth labels
            task_idx: the task index
            logits: the logits of the old network

        Returns:
            the differentiable loss value
        """

        outputs = self.net(inputs)[:, :self.n_seen_classes]
        if task_idx == 0:
            # Compute loss on the current task
            targets = self.eye[labels][:, :self.n_seen_classes]
            loss = F.binary_cross_entropy_with_logits(outputs, targets)
            assert loss >= 0
        else:
            targets = self.eye[labels][:, self.n_past_classes:self.n_seen_classes]
            comb_targets = torch.cat((logits[:, :self.n_past_classes], targets), dim=1)
            loss = F.binary_cross_entropy_with_logits(outputs, comb_targets)
            assert loss >= 0

        return loss

    def begin_task(self, dataset):
        self.noval_data_loader = dataset.train_loader
        icarl_replay(self, dataset)
        if self.current_task > 0:
            self.gram_schmidt_extend(self.dataset.N_CLASSES_PER_TASK)
            self._init_etf()
            print("The ETF shape is: ", self.etf.shape)
        

    def end_task(self, dataset) -> None:
        self.old_net = deepcopy(self.net.eval())
        self.net.train()
        with torch.no_grad():
            fill_buffer(self.buffer, dataset, self.current_task, net=self.net, use_herding=False)
        self.class_means = None

        if self.current_task == 0:
            self.noval_class_mean = self.get_noval_classmen()
            self.get_simi_orth()
            self.closest_orth_matrix()
            self._init_etf()

    @torch.no_grad()
    def compute_class_means(self) -> None:
        """
        Computes a vector representing mean features for each class.
        """
        # This function caches class means
        transform = self.dataset.get_normalization_transform()
        class_means = []
        buf_data = self.buffer.get_all_data(transform, device=self.device)
        examples, labels = buf_data[0], buf_data[1]
        for _y in self.classes_so_far:
            x_buf = torch.stack(
                [examples[i]
                 for i in range(0, len(examples))
                 if labels[i].cpu() == _y]
            ).to(self.device)
            with bn_track_stats(self, False):
                allt = None
                while len(x_buf):
                    batch = x_buf[:self.args.batch_size]
                    x_buf = x_buf[self.args.batch_size:]
                    feats = self.net(batch, returnt='features').mean(0)
                    if allt is None:
                        allt = feats
                    else:
                        allt += feats
                        allt /= 2
                class_means.append(allt.flatten())
        self.class_means = torch.stack(class_means)

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
        scaling_factor = math.sqrt(self.dataset.N_CLASSES_PER_TASK / (self.dataset.N_CLASSES_PER_TASK - 1))
        norm_center_class_mean = F.normalize(self.noval_class_mean - torch.mean(self.noval_class_mean, dim = 0), dim=1)
        intermediate = norm_center_class_mean.T / scaling_factor

        # Step 2: Compute the transformation matrix M
        i_nc_nc = torch.eye(self.dataset.N_CLASSES_PER_TASK)  # Identity matrix of size ac x ac
        one_nc_nc = torch.ones(self.dataset.N_CLASSES_PER_TASK, self.dataset.N_CLASSES_PER_TASK) / self.dataset.N_CLASSES_PER_TASK  # Matrix of ones scaled by 1/ac
        M = i_nc_nc - one_nc_nc  # Transformation matrix M

        # Step 3: Apply the pseudo-inverse of M to reverse the transformation
        M_inv = torch.linalg.pinv(M).to(self.device)  # Pseudo-inverse for numerical stability
        orth_recovered = torch.matmul(intermediate, M_inv)
        self.orth = orth_recovered

    def closest_orth_matrix(self) -> None:
        U, _, Vt = torch.linalg.svd(self.orth, full_matrices=False)
        Q = torch.matmul(U, Vt).to('cpu')
        self.orth = Q.numpy()


    # def _generate_random_orthogonal_matrix(self) -> None:
    #     rand_mat = np.random.random(size=(512, self.base_num))
    #     orth_vec, _ = np.linalg.qr(rand_mat)
    #     self.orth = orth_vec

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
        """
        Extend matrix A by num_new_cols orthogonal columns using Gram-Schmidt.
        """
        # Number of rows and current number of columns in A
        rows, cols = self.orth.shape

        # Generate random new vectors (columns)
        new_vectors = np.random.randn(rows, extend_num)

        # Orthogonalize new vectors with respect to A
        for i in range(extend_num):
            # Get the current new vector
            v = new_vectors[:, i]

            # Subtract projections onto the existing orthogonal columns of A and new vectors
            for j in range(cols):  # Project onto existing columns of A
                v -= np.dot(self.orth[:, j], v) * self.orth[:, j]
            for j in range(i):  # Project onto already orthogonalized new vectors
                v -= np.dot(new_vectors[:, j], v) * new_vectors[:, j]

            # Normalize the vector
            v /= np.linalg.norm(v)

            # Place it back in the new_vectors matrix
            new_vectors[:, i] = v

        # Stack the original matrix A and the new orthogonal vectors
        orth_extended = np.hstack([self.orth, new_vectors])

        self.orth = orth_extended
