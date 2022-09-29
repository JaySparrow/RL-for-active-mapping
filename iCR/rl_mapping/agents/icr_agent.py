import numpy as np
import os
import yaml
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint
from bc_exploration.footprints.footprints import CustomFootprint

from rl_mapping.utilities.utils_landmarks import R_matrix, f, grad_exp_u_hat, exp_u_hat, state_to_T, df_dx, df_du, unicycle_dyn
from rl_mapping.utilities.fov_utils import Gaussian_CDF, Gaussian_PDF, circle_SDF, triangle_SDF


def create_icr_landmarks_agent_from_params(params_filename, position_cov, num_landmarks, horizon, tau):

    assert os.path.exists(params_filename)
    with open(os.path.join(params_filename)) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    if params['footprint']['type'] == 'tricky_circle':
        footprint_points = get_tricky_circular_footprint()
    elif params['footprint']['type'] == 'tricky_oval':
        footprint_points = get_tricky_oval_footprint()
    elif params['footprint']['type'] == 'circle':
        rotation_angles = np.arange(0, 2 * np.pi, 4 * np.pi / 180)
        footprint_points = \
            params['footprint']['radius'] * np.array([np.cos(rotation_angles), np.sin(rotation_angles)]).T
    elif params['footprint']['type'] == 'pixel':
        footprint_points = np.array([[0., 0.]])
    else:
        footprint_points = None
        assert False and "footprint type specified not supported."

    footprint = CustomFootprint(footprint_points=footprint_points,
                                angular_resolution=params['footprint']['angular_resolution'],
                                inflation_scale=params['footprint']['inflation_scale'])

    num_iter = params['num_iter']
    kappa = params['kappa']
    v_0 = params['v_0']
    alpha = np.array([params['alpha']['_1'], params['alpha']['_2']])

    icr_landmark_agent = Icr(footprint=footprint,
                           horizon=horizon,
                           tau=tau,
                           num_iter=num_iter,
                           position_cov=position_cov,
                           num_landmarks=num_landmarks,
                           kappa=kappa,
                           v_0=v_0,
                           alpha=alpha)

    return icr_landmark_agent


class Icr:
    def __init__(self, footprint, horizon, tau, num_iter, position_cov, num_landmarks,
                 kappa, v_0, alpha):
        self._footprint = footprint
        self._horizon = horizon
        self.tau = tau
        self._num_iter = num_iter
        self._inv_position_cov = np.linalg.inv(position_cov)
        self._num_landmarks = num_landmarks
        self._kappa = kappa
        self._v_0 = v_0
        self._alpha = alpha
        self.radius = 2  # same as in the proposed method

        self._b = np.concatenate((np.zeros(3), np.tile(np.array([1, 0, 1]), self._num_landmarks)))

        self._vector_inds = (np.array([[0, 0], [0, 1], [1, 1]])[None,:,:] +
                             2 * np.arange(num_landmarks)[:,None,None]).reshape((3 * num_landmarks, 2))

    def get_footprint(self):
        return self._footprint

    def plan(self, robot_pose, mu_landmarks, sigma_landmarks, mode='icr_lqr'):
        print('Planning started...')

        u = self._initialize_motion()

        if mode is 'random':
            return u
        else:
            landmarks = mu_landmarks.reshape((self._num_landmarks, 2))
            robot_traj, sigma_list, u = self._iCR(robot_pose, landmarks, sigma_landmarks, u)
            if mode is 'icr':
                return u
            else:
                print('Variable \'mode\' should be one of the following: \'random\' / \'icr\' / \'icr_lqr\' ')

    def _iCR(self, robot_pose, landmarks, sigma_landmarks, u):
        print("iCR Path planning started...")

        robot_traj = np.zeros((3, self._horizon + 1))
        robot_traj[:, 0] = robot_pose

        for it in range(self._num_iter):
            sigma = sigma_landmarks.copy()
            inv_sigma = np.linalg.inv(sigma)

            # Forward pass to evaluate reward
            for k in range(self._horizon):
                robot_traj[:, k + 1] = unicycle_dyn(robot_traj[:, k], u[:, k], self.tau).astype(np.float32)
                inv_sigma += self._compute_M(robot_traj[:, k + 1], landmarks)

            sigma = np.linalg.inv(inv_sigma)

            # Gradient backpropagation
            for k in range(self._horizon - 1, -1, -1):
                T_k = state_to_T(robot_traj[:, k])
                grad_exp_0, grad_exp_1, grad_exp_2 = grad_exp_u_hat(u[:, k], self.tau)
                Lambda_s_0, Lambda_s_1, Lambda_s_2 = T_k @ grad_exp_0, T_k @ grad_exp_1, T_k @ grad_exp_2

                dJ_du_0 = 0
                dJ_du_1 = 0
                dJ_du_2 = 0

                for s in range(k + 1, self._horizon + 1):
                    landmarks_h = np.hstack((landmarks, np.ones((self._num_landmarks, 1))))
                    T_s = state_to_T(robot_traj[:, s])
                    q_s_h = np.linalg.inv(T_s) @ landmarks_h.T
                    d, dd_dq_s = circle_SDF(q_s_h.T[:, :2], self.radius)
                    inv_T_s = np.linalg.inv(T_s)
                    dq_s_du_0 = - inv_T_s[:2, :] @ Lambda_s_0 @ q_s_h
                    dq_s_du_1 = - inv_T_s[:2, :] @ Lambda_s_1 @ q_s_h
                    dq_s_du_2 = - inv_T_s[:2, :] @ Lambda_s_2 @ q_s_h
                    Phi = Gaussian_CDF(d, self._kappa)
                    Phi_der = Gaussian_PDF(d, self._kappa)

                    w_1_0 = - np.diag((Phi_der * np.sum(dd_dq_s * dq_s_du_0.T, axis=1)).repeat(2))
                    w_1_1 = - np.diag((Phi_der * np.sum(dd_dq_s * dq_s_du_1.T, axis=1)).repeat(2))
                    w_1_2 = - np.diag((Phi_der * np.sum(dd_dq_s * dq_s_du_2.T, axis=1)).repeat(2))

                    m_1 = np.kron(np.eye(self._num_landmarks), T_s[:2, :2] @ self._inv_position_cov @ T_s[:2, :2].T)

                    term_1_0 = w_1_0 @ m_1
                    term_1_1 = w_1_1 @ m_1
                    term_1_2 = w_1_2 @ m_1

                    w_2 = np.diag((1 - Phi).repeat(2))

                    m_2_0 = Lambda_s_0[:2, :2] @ self._inv_position_cov @ T_s[:2, :2].T
                    m_2_0 = np.kron(np.eye(self._num_landmarks), m_2_0 + m_2_0.T)
                    m_2_1 = Lambda_s_1[:2, :2] @ self._inv_position_cov @ T_s[:2, :2].T
                    m_2_1 = np.kron(np.eye(self._num_landmarks), m_2_1 + m_2_1.T)
                    m_2_2 = Lambda_s_2[:2, :2] @ self._inv_position_cov @ T_s[:2, :2].T
                    m_2_2 = np.kron(np.eye(self._num_landmarks), m_2_2 + m_2_2.T)

                    term_2_0 = w_2 @ m_2_0
                    term_2_1 = w_2 @ m_2_1
                    term_2_2 = w_2 @ m_2_2

                    dM_du_0 = term_1_0 + term_2_0
                    dM_du_1 = term_1_1 + term_2_1
                    dM_du_2 = term_1_2 + term_2_2

                    dJ_du_0 -= np.trace(sigma @ dM_du_0 @ sigma)
                    dJ_du_1 -= np.trace(sigma @ dM_du_1 @ sigma)
                    dJ_du_2 -= np.trace(sigma @ dM_du_2 @ sigma)

                    exp_u = exp_u_hat(u[:, s - 1], self.tau)
                    Lambda_s_0, Lambda_s_1, Lambda_s_2 = Lambda_s_0 @ exp_u, Lambda_s_1 @ exp_u, Lambda_s_2 @ exp_u

                u[0, k] -= self._alpha[0] * dJ_du_0
                u[1, k] -= self._alpha[1] * dJ_du_1
                #u[2, k] -= self._alpha[2] * dJ_du_2
                u[2, k] = 0
                

        sigma_list = [sigma_landmarks.copy()]
        inv_sigma = np.linalg.inv(sigma_list[-1])
        for k in range(self._horizon):
            robot_traj[:, k + 1] = f(robot_traj[:, k], u[:, k], np.zeros(3), self.tau)
            inv_sigma += self._compute_M(robot_traj[:, k + 1], landmarks)
            sigma_list.append(np.linalg.inv(inv_sigma))

        print("iCR path planning finished!\nProceeding to LQR gain computation...")
        return robot_traj, sigma_list, u

    def _initialize_motion(self):
        u = np.zeros((3, self._horizon))
        u[0, :] = np.random.uniform(low=-self._v_0, high=self._v_0, size=self._horizon)
        u[1, :] = np.random.uniform(low=-self._v_0, high=self._v_0, size=self._horizon)
        u[2, :] = np.zeros(self._horizon)
        return u

    def _compute_M(self, robot_pose, landmarks):
        R = R_matrix(robot_pose[2])
        q = (landmarks - robot_pose[:2]) @ R
        sdf, _ = circle_SDF(q, self.radius)
        Phi = Gaussian_CDF(sdf, self._kappa)
        diag_rotated_cov = np.kron(np.eye(self._num_landmarks), R @ self._inv_position_cov @ R.T)
        return np.diag((1 - Phi).repeat(2)) @ diag_rotated_cov
