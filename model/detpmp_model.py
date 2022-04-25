import numpy as np
import torch as th

class DeterministicProMP:

    def __init__(self, n_basis, n_dof, width=None, off=0.1, zero_start=False, n_zero_bases=0, step_length=None,
                 dt=0.02, weight_scale=1, before_traj_steps = 0):
        self.n_basis = n_basis
        self.n_dof = n_dof
        self.weights = th.zeros(size=(self.n_basis, self.n_dof))
        self.weight_scale = weight_scale
        if zero_start or before_traj_steps:
            self.n_zero_bases = n_zero_bases
        else:
            self.n_zero_bases = 0

        self.centers = np.linspace(-off, 1. + off, self.n_basis + self.n_zero_bases)
        if width is None:
            self.widths = np.ones(self.n_basis + self.n_zero_bases) * ((1. + off) / (2. * (self.n_basis + self.n_zero_bases)))
        else:
            self.widths = np.ones(self.n_basis + self.n_zero_bases) * width

        N = step_length
        t = np.linspace(0, 1, N)
        self.cr_scale = th.Tensor([step_length * dt]).to(device="cuda")
        self.t = th.Tensor(t).cuda()

        self.pos_features_np , self.vel_features_np , self.acc_features_np = self._exponential_kernel(t)
        self.pos_features = th.Tensor(self.pos_features_np[:, self.n_zero_bases:]).to(device="cuda")
        self.vel_features = th.Tensor(self.vel_features_np[:, self.n_zero_bases:]).to(device="cuda")
        self.acc_features = th.Tensor(self.acc_features_np[:, self.n_zero_bases:]).to(device="cuda")
        self.pos_features.requires_grad = True
        self.vel_features.requires_grad = True

    def initial_weights(self, initial_weights):
        self.weights = initial_weights * self.weight_scale

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


    def compute_trajectory(self):
        if self.weights.requires_grad == False:
            self.weights.requires_grad = True
        return self.t * self.cr_scale, th.matmul(self.pos_features, self.weights), \
               th.matmul(self.vel_features, self.weights) / self.cr_scale, \
               th.matmul(self.acc_features, self.weights) / th.square(self.cr_scale)
