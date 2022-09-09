import numpy as np


def state_to_T(state):
    return np.array([[np.cos(state[2]), -np.sin(state[2]), state[0]], [np.sin(state[2]), np.cos(state[2]), state[1]],
                     [0, 0, 1]])


def T_to_state(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def R_matrix(theta):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def f(x, u, w, tau):
    omega_tau_2 = u[1] * tau / 2
    v_sinc = u[0] * np.sinc(omega_tau_2 / np.pi)
    x_next = x + tau * np.array([v_sinc * np.cos(x[2] + omega_tau_2),
                                 v_sinc * np.sin(x[2] + omega_tau_2),
                                 u[1]]) + w
    return x_next


def df_dx(x, u, tau):
    a = u[1] * tau / 2
    f_1 = np.sinc(a / np.pi) * np.cos(a + x[2])
    f_2 = np.sinc(a / np.pi) * np.sin(a + x[2])
    return np.eye(3) + u[0] * tau * np.array([[0, 0, -f_2],
                                              [0, 0, f_1],
                                              [0, 0, 0]])


def df_du(x, u, tau):
    a = u[1] * tau / 2
    f_1 = np.sinc(a / np.pi) * np.cos(a + x[2])
    f_2 = np.sinc(a / np.pi) * np.sin(a + x[2])
    f_1_p = (np.cos(a) - np.sinc(a / np.pi)) * np.cos(x[2] + a) / a - np.sinc(a / np.pi) * np.sin(x[2] + a)
    f_2_p = (np.cos(a) - np.sinc(a / np.pi)) * np.sin(x[2] + a) / a + np.sinc(a / np.pi) * np.cos(x[2] + a)
    return tau * np.array([[f_1, u[0] * tau * f_1_p / 2],
                           [f_2, u[0] * tau * f_2_p / 2],
                           [0, 1]])


def dh_dx(x, y):
    R_1 = R_matrix(x[2] + np.pi / 2).T
    third_col = R_1 @ (y - x[:2]).T
    third_col = third_col.reshape((2 * third_col.shape[1], 1), order='F')
    R_2 = R_matrix(np.pi - x[2])
    first_two_cols = np.tile(R_2, (int(third_col.shape[0] / 2), 1))
    return np.hstack((first_two_cols, third_col))


def dh_dy(x):
    return R_matrix(-x[2])


# def grad_exp_u_hat(u, tau):
#     sin_t, cos_t = np.sin(u[1] * tau), np.cos(u[1] * tau)
#     if u[1] == 0:
#         elem_1, elem_2, elem_3, elem_4 = tau, tau, -u[1] * tau**2 / 2, u[0] * tau**2 / 2
#     else:
#         elem_1, elem_2 = sin_t / u[1], (1 - cos_t) / u[1]
#         elem_3 = u[0] * (u[1] * tau * cos_t - sin_t) / u[1]**2
#         elem_4 = u[0] * (u[1] * tau * sin_t - (1 - cos_t)) / u[1]**2
#
#     grad_exp_0 = np.array([[0, 0, elem_1], [0, 0, elem_2], [0, 0, 0]])
#     grad_exp_1 = np.array([[-tau * sin_t, -tau * cos_t, elem_3], [tau * cos_t, -tau * sin_t, elem_4], [0, 0, 0]])
#     return grad_exp_0, grad_exp_1


def exp_u_hat(u, tau):
    u_hat = np.array([[0, -u[1], u[0]], [u[1], 0, 0], [0, 0, 0]])
    u_hat_sq = u_hat @ u_hat
    if u[1] == 0:
        exp_uhat = np.identity(3) + tau * u_hat + tau**2 * u_hat_sq / 2
    else:
        exp_uhat = np.identity(3) + np.sin(u[1] * tau) / u[1] * u_hat + (1 - np.cos(u[1] * tau)) / u[1]**2 * u_hat_sq
    return exp_uhat

def grad_exp_u_hat(u, tau):
    omega_tau = u[2] * tau
    sin_t, cos_t = np.sin(omega_tau), np.cos(omega_tau)
    if u[2] == 0:
      f_1, f_2, f_1_der, f_2_der = 1, 0, 0, 1/2
    else:
      f_1, f_2 = sin_t / omega_tau, (1 - cos_t) / omega_tau
      f_1_der = (omega_tau * cos_t - sin_t) / (omega_tau**2)
      f_2_der = (omega_tau * sin_t - (1 - cos_t)) / (omega_tau**2)
    grad_exp_0 = np.array([[0, 0, tau * f_1], [0, 0, tau * f_2], [0, 0, 0]])
    grad_exp_1 = np.array([[0, 0, - tau * f_2], [0, 0, tau * f_1], [0, 0, 0]])
    grad_exp_2 = tau * np.array([[-sin_t, -cos_t, tau * (u[0] * f_1_der - u[1] * f_2_der)], [cos_t, - sin_t, tau * (u[1] * f_1_der + u[0] * f_2_der)], [0, 0, 0]])
    return grad_exp_0, grad_exp_1, grad_exp_2

def SE2_motion(T, u, dt):
    exp_uhat = exp_hat(u,dt)
    T_next = T @ exp_uhat
    return T_next

def exp_hat(u, dt):
    u_hat = np.array([[0, -u[2], u[0]], [u[2], 0, u[1]], [0, 0, 0]])
    u_hat_sq = u_hat @ u_hat
    if u[2] == 0:
        exp_uhat = np.identity(3) + dt * u_hat + dt**2 * u_hat_sq / 2
    else:
        exp_uhat = np.identity(3) + np.sin(u[2] * dt) / u[2] * u_hat + (1 - np.cos(u[2] * dt)) / u[2]**2 * u_hat_sq
    return exp_uhat

def unicycle_dyn(state,u,dt):
    T = state_to_T(state)
    T_next = SE2_motion(T,u,dt)
    state_next = T_to_state(T_next) + np.random.normal(0, .2, [3,])
    return state_next
