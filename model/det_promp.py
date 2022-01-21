import numpy as np
import matplotlib.pyplot as plt
import torch as th

class DeterministicProMP:

    def __init__(self, n_basis, n_dof, width=None, off=0.2, zero_start=False, zero_goal=False, n_zero_bases=2, step_length=None, dt=0.02):
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.weights = np.zeros(shape=(self.n_basis, self.n_dof))
        self.n_zero_bases = n_zero_bases
        #add_basis = 0
        add_basis = n_zero_bases
        #if zero_goal:
        #    add_basis += n_zero_bases
        self.centers = np.linspace(-off, 1. + off, self.n_basis + add_basis)
        if width is None:
            self.widths = np.ones(self.n_basis + add_basis) * ((1. + off) / (2. * (self.n_basis + add_basis)))
        else:
            self.widths = np.ones(self.n_basis + add_basis) * width
        self.zero_start = zero_start
        self.zero_goal = zero_goal

        self.step_length = step_length
        self.corrected_scale = self.step_length * dt

        N = self.step_length
        t = np.linspace(0, 1, N)
        self.pos_features_np , self.vel_features_np , self.acc_features_np = self._exponential_kernel(t)

        #self.pos_features_np = self.pos_features_np[:, self.n_zero_bases:]
        #self.vel_features_np = self.vel_features_np[:, self.n_zero_bases:]
        #self.acc_features_np = self.acc_features_np[:, self.n_zero_bases:]
        self.pos_features_np = self.pos_features_np[:, self.n_zero_bases:]
        #print("sjape", self.pos_features_np.shape)
        self.vel_features_np = self.vel_features_np[:, self.n_zero_bases:]
        self.acc_features_np = self.acc_features_np[:, self.n_zero_bases:]

        self.t_np = t

        self.pos_features = th.Tensor(self.pos_features_np).to(device="cuda")
        self.vel_features = th.Tensor(self.vel_features_np).to(device="cuda")
        self.acc_features = th.Tensor(self.acc_features_np).to(device="cuda")
        self.cr_scale = th.Tensor([self.corrected_scale]).to(device="cuda")
        self.t = th.Tensor(t).cuda()

    def _exponential_kernel(self, z):
        z_ext = z[:, None]
        diffs = z_ext - self.centers[None, :]
        w = np.exp(-(np.square(diffs) / (2 * self.widths[None, :])))
        w_der = -(diffs / self.widths[None, :]) * w
        w_der2 = -(1 / self.widths[None, :]) * w + np.square(diffs / self.widths[None, :]) * w
        sum_w = np.sum(w, axis=1)[:, None]
        sum_w_der = np.sum(w_der, axis=1)[:, None]
        sum_w_der2 = np.sum(w_der2, axis=1)[:, None]
        tmp = w_der * sum_w - w * sum_w_der
        return w / sum_w, tmp / np.square(sum_w), \
               ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / np.power(sum_w, 3)


    def compute_trajectory(self, weights):
        if self.weights.requires_grad == False:
            self.weights.requires_grad = True
        return self.t * self.cr_scale, th.matmul(self.pos_features, weights), \
               th.matmul(self.vel_features, weights) / self.cr_scale, \
               th.matmul(self.acc_features, weights) / th.square(self.cr_scale)

    def compute_trajectory_with_noise(self, weights):
        return self.t_np * self.corrected_scale, np.dot(self.pos_features_np, weights), \
               np.dot(self.vel_features_np, weights) / self.corrected_scale, \
               np.dot(self.acc_features_np, weights) / np.square(self.corrected_scale)