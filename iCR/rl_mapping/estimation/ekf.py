import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from rl_mapping.utilities.utils_landmarks import f, df_dx, dh_dx, dh_dy, R_matrix


class EKF:
    def __init__(self, num_landmarks, motion_model_cov, position_cov, tau, init_robot_pose, init_landmark_pose):
        self._mu = np.zeros(3 + 2 * num_landmarks)
        self._mu[:3] = init_robot_pose
        self._mu[3:] = init_landmark_pose.reshape(2 * num_landmarks)
        self._sigma = np.diag(np.ones(3 + 2 * num_landmarks))

        self._n_l = num_landmarks

        self._motion_model_cov = motion_model_cov
        self._position_cov = position_cov

        self._tau = tau

    def predict(self, u):
        self._mu[:3] = f(self._mu[:3], u, np.zeros(3), self._tau)
        A = np.block([[df_dx(self._mu[:3], u, self._tau), np.zeros((3, 2 * self._n_l))],
                      [np.zeros((2 * self._n_l, 3)), np.diag(np.ones(2 * self._n_l))]])
        self._sigma = A @ self._sigma @ A.T
        self._sigma[:3, :3] += self._motion_model_cov

        return self._mu.copy(), self._sigma.copy()

    def update(self, z):
        visible_inds = np.nonzero(z[:, 0])[0]
        if visible_inds.shape[0] == 0:
            return self._mu, self._sigma

        visible_z = z[visible_inds, :]
        y = self._mu[3:].reshape((self._n_l, 2))
        visible_y = y[visible_inds, :]
        H_1 = dh_dx(self._mu[:3], visible_y)
        H_2 = np.kron(np.eye(visible_y.shape[0]), dh_dy(self._mu[:3]))
        H = np.hstack((H_1, H_2))
        V = np.kron(np.eye(visible_y.shape[0]), self._position_cov)

        visible_inds_rep = (2 * visible_inds + 3).repeat(2)
        visible_inds_rep[1::2] += 1
        update_inds = np.concatenate((np.arange(3), visible_inds_rep))
        sigma_mask = np.ix_(update_inds, update_inds)
        visible_sigma = self._sigma[sigma_mask]

        L = visible_sigma @ H.T @ np.linalg.inv(H @ visible_sigma @ H.T + V)

        h = (R_matrix(-self._mu[2]) @ (visible_y - self._mu[:2]).T).reshape(2 * visible_inds.shape[0], order='F')

        self._mu[update_inds] += L @ (visible_z.reshape(2 * visible_inds.shape[0]) - h)
        self._sigma[sigma_mask] = (np.eye(2 * visible_inds.shape[0] + 3) - L @ H) @ visible_sigma

        return self._mu.copy(), self._sigma.copy()

    def get_state_est(self):
        return self._mu.copy(), self._sigma.copy()
