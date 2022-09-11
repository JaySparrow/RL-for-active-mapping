import numpy as np

from scipy.special import erf


def triangle_SDF(q, psi, r):
    x, y = q[:, 0], q[:, 1]
    p_x = r / (1 + np.sin(psi))

    a_1, a_2, a_3 = np.array([-1, 1 / np.tan(psi)]), np.array([-1, -1 / np.tan(psi)]), np.array([1, 0])
    b_1, b_2, b_3 = 0, 0, -r
    q_1, q_2, q_3 = np.array([r, r * np.tan(psi)]), np.array([r, -r * np.tan(psi)]), np.array([0, 0])
    l_1_low, l_1_up, l_2_low, l_2_up = l_function(x, psi, r, p_x)

    SDF = np.linalg.norm(q - q_2, axis=1)
    SDF_grad = (q - q_2) / SDF[:, None]

    P_1_inds = np.nonzero(y >= l_1_up)[0]
    SDF[P_1_inds] = np.linalg.norm(q[P_1_inds, :] - q_1, axis=-1)
    SDF_grad[P_1_inds] = (q[P_1_inds, :] - q_1) / SDF[P_1_inds, None]

    D_1_inds = np.nonzero(np.logical_and(l_1_low <= y, y < l_1_up))[0]
    SDF[D_1_inds] = (q[D_1_inds, :] @ a_1 + b_1) / np.linalg.norm(a_1)
    SDF_grad[D_1_inds] = a_1 / np.linalg.norm(a_1)

    P_3_inds = np.nonzero(np.logical_and(x < 0, np.logical_and(l_2_up <= y, y < l_1_low)))[0]
    SDF[P_3_inds] = np.linalg.norm(q[P_3_inds, :] - q_3, axis=-1)
    SDF_grad[P_3_inds] = (q[P_3_inds, :] - q_3) / SDF[P_3_inds, None]

    D_3_inds = np.nonzero(np.logical_and(x > p_x, np.logical_and(l_2_up <= y, y < l_1_low)))[0]
    SDF[D_3_inds] = (q[D_3_inds, :] @ a_3 + b_3) / np.linalg.norm(a_3)
    SDF_grad[D_3_inds] = a_3 / np.linalg.norm(a_3)

    D_2_inds = np.nonzero(np.logical_and(l_2_low < y, y < l_2_up))[0]
    SDF[D_2_inds] = (q[D_2_inds, :] @ a_2 + b_2) / np.linalg.norm(a_2)
    SDF_grad[D_2_inds] = a_2 / np.linalg.norm(a_2)

    return SDF, SDF_grad


def l_function(x, psi, r, p_x):
    l_1_low, l_2_up = r * np.tan(psi) * np.ones(x.shape), -r * np.tan(psi) * np.ones(x.shape)

    inds_1 = np.nonzero(x < 0)
    l_1_low[inds_1], l_2_up[inds_1] = - x[inds_1] / np.tan(psi), x[inds_1] / np.tan(psi)

    inds_2 = np.nonzero(np.logical_and(0 <= x, x < p_x))
    l_1_low[inds_2], l_2_up[inds_2] = 0, 0

    inds_3 = np.nonzero(np.logical_and(p_x <= x, x < r))
    l_1_low[inds_3] = np.tan(np.pi / 4 + psi / 2) * x[inds_3] - r / np.cos(psi)
    l_2_up[inds_3] = - np.tan(np.pi / 4 + psi / 2) * x[inds_3] + r / np.cos(psi)

    l_1_up, l_2_low = r * np.tan(psi) * np.ones(x.shape), -r * np.tan(psi) * np.ones(x.shape)

    inds_4 = np.nonzero(x < r)
    l_1_up[inds_4] = - (x[inds_4] - r) / np.tan(psi) + r * np.tan(psi)
    l_2_low[inds_4] = (x[inds_4] - r) / np.tan(psi) - r * np.tan(psi)

    return l_1_low, l_1_up, l_2_low, l_2_up


def Gaussian_CDF(x, kap):
    Phi = (1 + erf(x / (np.sqrt(2) * kap) - 2)) / 2
    return Phi


def Gaussian_PDF(x, kap):
    Phi_der = 1 / (np.sqrt(2 * np.pi) * kap) * np.exp(- (x / (np.sqrt(2) * kap) - 2) ** 2)
    return Phi_der

def circle_SDF(q, r):
    SDF, Grad = np.linalg.norm(q, axis=1) - r, 2 * q
    return SDF, Grad

# import numpy as np
# def main():
#     print(triangle_SDF(np.array([[2,3], [3,5], [2, 8]]), .5, 0.5))
#
# if __name__ == '__main__':
#     main()