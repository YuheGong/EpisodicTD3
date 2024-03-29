import numpy as np
import torch as th

class DeterministicProMP:

    def __init__(self, n_basis, n_dof, basis_function="rbf", width=None, off=0.2, zero_start=False, n_zero_bases=0, step_length=None,
                 dt=0.02, weight_scale=1, pos_traj_steps=0):

        self.n_basis = n_basis
        self.n_dof = n_dof
        self.weights = th.zeros(size=(self.n_basis, self.n_dof))
        self.weight_scale = weight_scale

        if zero_start:
            self.n_zero_bases = n_zero_bases
        else:
            self.n_zero_bases = 0

        self.centers = np.linspace(-off, 1. + off, self.n_basis + self.n_zero_bases)
        if width is None:
            self.widths = np.ones(self.n_basis + self.n_zero_bases) * ((1. + off) / (2. * (self.n_basis + self.n_zero_bases)))
        else:
            self.widths = np.ones(self.n_basis + self.n_zero_bases) * width

        # basis_function = "rythmic"

        if basis_function == "rbf":
            self._exponential_kernel = self._exponential_kernel_RBF
        elif basis_function == "rythmic":
            self._exponential_kernel = self._exponential_kernel_Rythmic

        N = step_length
        t = np.linspace(0, 1, N)
        self.cr_scale = th.Tensor([step_length * dt]).to(device="cuda")
        self.t = th.Tensor(t).cuda()

        self.pos_traj_steps = pos_traj_steps

        #drop out the featrues for zero_basis
        self.pos_features_np, self.vel_features_np , self.acc_features_np = self._exponential_kernel(t)

        if self.pos_traj_steps > 0:
            self.pos_features_np = np.vstack([self.pos_features_np, np.tile(self.pos_features_np[-1, :], [self.pos_traj_steps, 1])])
            #self.pos_features_np = np.vstack([self.pos_features_np, np.zeros(shape=(self.pos_traj_steps, self.n_basis + self.n_zero_bases))])
            self.vel_features_np = np.vstack([self.vel_features_np, np.zeros(shape=(self.pos_traj_steps, self.n_basis + self.n_zero_bases))])

        self.pos_features_np *= self.weight_scale
        self.vel_features_np *= self.weight_scale
        self.acc_features_np *= self.weight_scale
        self.pos_features_np = self.pos_features_np[:, self.n_zero_bases:]
        self.vel_features_np = self.vel_features_np[:, self.n_zero_bases:]
        self.acc_features_np = self.acc_features_np[:, self.n_zero_bases:]
        self.pos_features = th.Tensor(self.pos_features_np).to(device="cuda")
        self.vel_features = th.Tensor(self.vel_features_np).to(device="cuda")
        self.acc_features = th.Tensor(self.acc_features_np).to(device="cuda")
        self.pos_features.requires_grad = True
        self.vel_features.requires_grad = True

    def _exponential_kernel_RBF(self, z):
        z_ext = z[:, None]
        diffs = z_ext - self.centers[None, :]  ## broadcast z_ext to diffs, center for each steps
        w = np.exp(-(np.square(diffs) / (2 * self.widths[None, :])))  # stroke-based movements
        w_der = -(diffs / self.widths[None, :]) * w
        w_der2 = -(1 / self.widths[None, :]) * w + np.square(diffs / self.widths[None, :]) * w
        sum_w = np.sum(w, axis=1)[:, None]
        sum_w_der = np.sum(w_der, axis=1)[:, None]
        sum_w_der2 = np.sum(w_der2, axis=1)[:, None]
        tmp = w_der * sum_w - w * sum_w_der
        return w / sum_w, tmp / np.square(sum_w), \
               ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / np.power(sum_w, 3)

    def _exponential_kernel_Rythmic(self, z):
        z_ext = z[:, None]
        diffs = z_ext - self.centers[None, :]
        w_inside = diffs / self.widths[None, :] * 2 * np.pi
        w = np.cos(w_inside)
        w_der = -2 * np.pi / self.widths[None, :] * np.sin(w_inside)
        w_der2 = -np.square(2 * np.pi / self.widths[None, :]) * w
        sum_w = np.sum(w, axis=1)[:, None]
        sum_w_der = np.sum(w_der, axis=1)[:, None]
        sum_w_der2 = np.sum(w_der2, axis=1)[:, None]
        tmp = w_der * sum_w - w * sum_w_der
        return w / sum_w, tmp / np.square(sum_w), \
               ((w_der2 * sum_w - sum_w_der2 * w) * sum_w - 2 * sum_w_der * tmp) / np.power(sum_w, 3)

    def compute_trajectory(self):
        if self.weights.requires_grad == False:
            self.weights.requires_grad = True
        if self.pos_features.requires_grad == False:
            self.pos_features.requires_grad = True
        if self.vel_features.requires_grad == False:
            self.vel_features.requires_grad = True
        if self.weights.requires_grad == False:
            self.weights.requires_grad = True
        return self.t * self.cr_scale, th.matmul(self.pos_features, self.weights), \
               th.matmul(self.vel_features, self.weights) / self.cr_scale, \
               th.matmul(self.acc_features, self.weights) / th.square(self.cr_scale)
