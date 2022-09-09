import numpy as np
import cv2

from scipy.special import gamma, digamma, erf

from bc_exploration.utilities.util import xy_to_rc


np.random.seed(30)
COLORS = np.round(np.random.uniform(0, 255, (20, 3)))


def B_func(alpha):
    if len(alpha.shape) == 2:
        axis = 1
    else:
        axis = 0
    num = np.prod(gamma(alpha), axis=axis)
    denom = gamma(np.sum(alpha, axis=axis))
    return num / (denom + 1e-10)


def entropy(alpha):
    if len(alpha.shape) == 2:
        K = alpha.shape[1]
        axis = 1
    else:
        K = alpha.shape[0]
        axis = 0

    alpha_0 = np.sum(alpha, axis=axis)

    result = -1 * np.sum((alpha -1) * digamma(alpha), axis=axis)
    result += (alpha_0 - K) * digamma(alpha_0)
    result += np.log((B_func(alpha)))

    return result


def conditional_entropy(alpha):
    if len(alpha.shape) == 2:
        K = alpha.shape[1]
        axis = 1
    else:
        K = alpha.shape[0]
        axis = 0

    alpha_0 = np.sum(alpha, axis=axis)

    result = -1 * alpha * digamma(alpha + 1)
    result += (1 - alpha) * (K - 1) * digamma(alpha)

    for i in range(K):
        e = np.zeros(alpha.shape)
        e[:, i] = 1
        result[:, i] += alpha[:, i] * np.log(B_func(alpha + e))

    result = np.sum(result, axis=axis) / alpha_0
    result += (alpha_0 + 1 - K) * digamma(alpha_0 + 1)

    return result


def softmax(l):
    return np.exp(l) / np.sum(np.exp(l))


def visualize(semantic_map, num_class, render_size, wait_key, state=None, save_file=None):
    vis_map = semantic_map.copy()
    vis_map.data = cv2.cvtColor(vis_map.data, cv2.COLOR_GRAY2RGB)

    if num_class > 1:
        vis_map_shape = semantic_map.get_shape()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        for k in range(num_class):
            mask = np.zeros(vis_map_shape, dtype=np.uint8)
            mask[semantic_map.data == k + 1] = 1
            mask.data = cv2.dilate(mask, kernel, iterations=3)
            class_coords = np.nonzero(mask == 1)

            if len(class_coords[0]) > 0:

                vis_map.data[class_coords[0], class_coords[1], :] = COLORS[k, :]

    if state is not None:
        robot_rc = xy_to_rc(state[:2], vis_map).astype(np.int)

        cv2.circle(vis_map.data, (robot_rc[1], robot_rc[0]), 5, (255, 0, 0), -1)

        vis_map.data = cv2.arrowedLine(vis_map.data, (robot_rc[1], robot_rc[0]),
                                       (robot_rc[1] + np.round(20 * np.cos(state[2])).astype(np.int),
                                        robot_rc[0] - np.round(20 * np.sin(state[2])).astype(np.int)), (0, 0, 255), 2,
                                       tipLength = 0.5)

    if save_file is not None:
        assert isinstance(save_file, str)
        cv2.imwrite(save_file, vis_map.data)
        return
    else:
        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', vis_map.data)
        cv2.resizeWindow('map', *render_size)
        cv2.waitKey(1)


def visualize_info(info_map, render_size, wait_key, state=None, save_file=None):
    vis_map = info_map.copy()
    vis_map.data = (np.log(vis_map.data[:,:,1] + 1)).astype(np.uint8) * 35
    vis_map.data = cv2.cvtColor(vis_map.data, cv2.COLOR_GRAY2RGB)

    vis_map.data = cv2.applyColorMap(vis_map.data, cv2.COLORMAP_HOT)

    if state is not None:
        robot_rc = xy_to_rc(state[:2], vis_map).astype(np.int)

        cv2.circle(vis_map.data, (robot_rc[1], robot_rc[0]), 5, (255, 0, 0), -1)

        vis_map.data = cv2.arrowedLine(vis_map.data, (robot_rc[1], robot_rc[0]),
                                       (robot_rc[1] + np.round(20 * np.cos(state[2])).astype(np.int),
                                        robot_rc[0] - np.round(20 * np.sin(state[2])).astype(np.int)), (0, 0, 255), 2,
                                       tipLength = 0.5)

    if save_file is not None:
        assert isinstance(save_file, str)
        cv2.imwrite(save_file, vis_map.data)
        return
    else:
        cv2.namedWindow('info_map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('info_map', vis_map.data)
        cv2.resizeWindow('info_map', *render_size)
        cv2.waitKey(1)


def visualize_traj(semantic_map, num_class, render_size, state, old_traj, new_traj, wait_key, save_file=None):
    vis_map = semantic_map.copy()
    vis_map.data = cv2.cvtColor(vis_map.data, cv2.COLOR_GRAY2RGB)

    if num_class > 1:
        vis_map_shape = semantic_map.get_shape()
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        for k in range(num_class):
            mask = np.zeros(vis_map_shape, dtype=np.uint8)
            mask[semantic_map.data == k + 1] = 1
            mask.data = cv2.dilate(mask, kernel, iterations=3)
            class_coords = np.nonzero(mask == 1)

            if len(class_coords[0]) > 0:

                vis_map.data[class_coords[0], class_coords[1], :] = COLORS[k, :]

    robot_rc = xy_to_rc(state[:2], vis_map).astype(np.int)

    cv2.circle(vis_map.data, (robot_rc[1], robot_rc[0]), 5, (255, 0, 0), -1)

    vis_map.data = cv2.arrowedLine(vis_map.data, (robot_rc[1], robot_rc[0]),
                                   (robot_rc[1] + np.round(20 * np.cos(state[2])).astype(np.int),
                                    robot_rc[0] - np.round(20 * np.sin(state[2])).astype(np.int)), (255, 0, 0), 2,
                                   tipLength = 0.5)

    for (old_pose, new_pose) in zip(old_traj, new_traj):
        old_rc = xy_to_rc(old_pose[:2], vis_map).astype(np.int)
        new_rc = xy_to_rc(new_pose[:2], vis_map).astype(np.int)

        cv2.circle(vis_map.data, (old_rc[1], old_rc[0]), 5, (0, 255, 0), -1)
        cv2.circle(vis_map.data, (new_rc[1], new_rc[0]), 5, (0, 0, 255), -1)

        vis_map.data = cv2.arrowedLine(vis_map.data, (old_rc[1], old_rc[0]),
                                       (old_rc[1] + np.round(20 * np.cos(old_pose[2])).astype(np.int),
                                        old_rc[0] - np.round(20 * np.sin(old_pose[2])).astype(np.int)), (0, 255, 0), 2,
                                       tipLength=0.5)
        vis_map.data = cv2.arrowedLine(vis_map.data, (new_rc[1], new_rc[0]),
                                       (new_rc[1] + np.round(20 * np.cos(new_pose[2])).astype(np.int),
                                        new_rc[0] - np.round(20 * np.sin(new_pose[2])).astype(np.int)), (0, 0, 255), 2,
                                       tipLength=0.5)

    if save_file is not None:
        assert isinstance(save_file, str)
        cv2.imwrite(save_file, vis_map.data)
        return
    else:
        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', vis_map.data)
        cv2.resizeWindow('map', *render_size)
        cv2.waitKey(1)


def state_to_T(state):
    return np.array([[np.cos(state[2]), -np.sin(state[2]), state[0]], [np.sin(state[2]), np.cos(state[2]), state[1]],
                     [0, 0, 1]])


def T_to_state(T):
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


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


def Grad_exp_hat(u,dt):
    sin_t, cos_t = np.sin(u[2] * dt), np.cos(u[2] * dt)
    if u[2] == 0:
        elem_1, elem_2, elem_3, elem_4 = dt, dt, -u[1] * dt**2 / 2, u[0] * dt**2 / 2
    else:
        elem_1, elem_2 = sin_t / u[2], (1 - cos_t) / u[2]
        elem_3 = (u[0] * (u[2] * dt * cos_t - sin_t) - u[1] * (u[2] * dt * sin_t - (1 - cos_t))) / u[2]**2
        elem_4 = (u[1] * (u[2] * dt * cos_t - sin_t) + u[0] * (u[2] * dt * sin_t - (1 - cos_t))) / u[2]**2

    Grad_exp_1 = np.array([[0, 0, elem_1], [0, 0, elem_2], [0, 0, 0]])
    Grad_exp_2 = np.array([[0, 0, -elem_2], [0, 0, elem_1], [0, 0, 0]])
    Grad_exp_3 = np.array([[-dt * sin_t, -dt * cos_t, elem_3], [dt * cos_t, -dt * sin_t, elem_4], [0, 0, 0]])
    return Grad_exp_1, Grad_exp_2, Grad_exp_3


def l_function(x, psi, r, p_x):
    if x < 0:
        l_1_low, l_2_up = - x / np.tan(psi), x / np.tan(psi)
    elif 0 <= x < p_x:
        l_1_low, l_2_up = 0, 0
    elif p_x <= x < r:
        l_1_low = np.tan(np.pi / 4 + psi / 2) * x - r / np.cos(psi)
        l_2_up = - np.tan(np.pi / 4 + psi / 2) * x + r / np.cos(psi)
    else:
        l_1_low, l_2_up = r * np.tan(psi), -r * np.tan(psi)

    if x < r:
        l_1_up, l_2_low = - (x - r) / np.tan(psi) + r * np.tan(psi), (x - r) / np.tan(psi) - r * np.tan(psi)
    else:
        l_1_up, l_2_low = r * np.tan(psi), -r * np.tan(psi)
    return l_1_low, l_1_up, l_2_low, l_2_up


def triangle_SDF(q, psi, r):
    x, y = q[0], q[1]
    p_x = r / (1 + np.sin(psi))

    a_1, a_2, a_3 = np.array([-1, 1 / np.tan(psi)]), np.array([-1, -1 / np.tan(psi)]), np.array([1, 0])
    b_1, b_2, b_3 = 0, 0, -r
    q_1, q_2, q_3 = np.array([r, r * np.tan(psi)]), np.array([r, -r * np.tan(psi)]), np.array([0, 0])
    l_1_low, l_1_up, l_2_low, l_2_up = l_function(x, psi, r, p_x)
    if y >= l_1_up:
        # P_1
        SDF, Grad = np.linalg.norm(q - q_1), (q - q_1) / np.linalg.norm(q - q_1)
    elif l_1_low <= y < l_1_up:
        # D_1
        SDF, Grad = (a_1 @ q + b_1) / np.linalg.norm(a_1), a_1 / np.linalg.norm(a_1)
    elif x < 0 and l_2_up <= y < l_1_low:
        # P_3
        SDF, Grad = np.linalg.norm(q - q_3), (q - q_3) / np.linalg.norm(q - q_3)
    elif x > p_x and l_2_up <= y < l_1_low:
        # D_3
        SDF, Grad = (a_3 @ q + b_3) / np.linalg.norm(a_3), a_3 / np.linalg.norm(a_3)
    elif y < l_2_up and y > l_2_low:
        # D_2
        SDF, Grad = (a_2 @ q + b_2) / np.linalg.norm(a_2), a_2 / np.linalg.norm(a_2)
    else:
        # P_2
        SDF, Grad = np.linalg.norm(q - q_2), (q - q_2) / np.linalg.norm(q - q_2)
    return SDF, Grad

def circle_SDF(q, r):
    if q.ndim in {1, 2}:
        SDF, Grad = np.linalg.norm(q, axis=0) - r, 2 * q
    else:
        raise NotImplementedError
    return SDF, Grad

def Gaussian_CDF(x, kap):
    Psi = (1 + erf(x / (np.sqrt(2) * kap) - 2)) / 2
    Psi_der = 1 / (np.sqrt(2 * np.pi) * kap) * np.exp(- (x / (np.sqrt(2) * kap) - 2) ** 2)
    return Psi, Psi_der

def visualize_FOV(kappa, psi, r, res):
    x = np.linspace(-3, 6, res)
    y = np.linspace(-4, 4, res)
    FOV_surface = np.zeros((res, res, 3))
    for k, kap in enumerate([kappa / 100, kappa, kappa * 2]):
        for i in range(res):
            for j in range(res):
                q = np.array([x[i], y[j]])
                FOV_surface[i, j, k] = 1 - Gaussian_CDF(triangle_SDF(q, psi, r)[0], kap)[0]

    vis_map = cv2.cvtColor((FOV_surface[:, :, 1] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_HOT)

    contour_map = (FOV_surface[:, :, 2] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    contour_map = (FOV_surface[:, :, 1] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    contour_map = (FOV_surface[:, :, 0] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    # contour_map = (FOV_surface[:, :, 0] * 255).astype(np.uint8)
    # _, thresh = cv2.threshold(contour_map, 157, 255, 0)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(vis_map, contours, -1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    # _, thresh = cv2.threshold(contour_map, 97, 255, 0)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(vis_map, contours, -1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    #
    # contour_map = (FOV_surface[:, :, 2] * 255).astype(np.uint8)
    # _, thresh = cv2.threshold(contour_map, 157, 255, 0)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(vis_map, contours, -1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
    # _, thresh = cv2.threshold(contour_map, 97, 255, 0)
    # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(vis_map, contours, -1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite('FOV_surface2.png', vis_map)

def visualize_FOV_circle(kappa, r, res):
    x = np.linspace(-3, 6, res)
    y = np.linspace(-4, 4, res)
    FOV_surface = np.zeros((res, res, 3))
    for k, kap in enumerate([kappa / 100, kappa, kappa * 2]):
        for i in range(res):
            for j in range(res):
                q = np.array([x[i], y[j]])
                FOV_surface[i, j, k] = 1 - Gaussian_CDF(circle_SDF(q, r)[0], kap)[0]

    vis_map = cv2.cvtColor((FOV_surface[:, :, 1] * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    vis_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_HOT)

    contour_map = (FOV_surface[:, :, 2] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(255, 0, 0), thickness=1, lineType=cv2.LINE_AA)

    contour_map = (FOV_surface[:, :, 1] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    contour_map = (FOV_surface[:, :, 0] * 255).astype(np.uint8)
    _, thresh = cv2.threshold(contour_map, 250, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_map, contours, -1, color=(255, 255, 0), thickness=1, lineType=cv2.LINE_AA)

    cv2.imwrite('FOV_surface2.png', vis_map)

# visualize_FOV(0.5, np.pi/10, 3, 1000)
