import numpy as np
import cv2

from bc_exploration.utilities.util import xy_to_rc
from bc_exploration.mapping.costmap import Costmap


def visualize(robot_pose_gt, robot_pose_est, landmarks_gt, landmarks_est, state_covariance,
              map_size, map_res, map_origin, render_size, trajectory=None, save_file=None):
    vis_map = Costmap(data=cv2.cvtColor((Costmap.FREE * np.ones(map_size)).astype(np.uint8),
                                        cv2.COLOR_GRAY2RGB), resolution=map_res, origin=map_origin)

    robot_rc_gt = xy_to_rc(robot_pose_gt, vis_map).astype(int)
    cv2.circle(vis_map.data, (robot_rc_gt[1], robot_rc_gt[0]), 5, (0, 255, 0), -1)
    vis_map.data = cv2.arrowedLine(vis_map.data, (robot_rc_gt[1], robot_rc_gt[0]),
                                   (robot_rc_gt[1] + np.round(20 * np.cos(robot_pose_gt[2])).astype(np.int),
                                    robot_rc_gt[0] - np.round(20 * np.sin(robot_pose_gt[2])).astype(np.int)),
                                   (0, 255, 0), 2, tipLength=0.5)

    robot_rc_est = xy_to_rc(robot_pose_est, vis_map).astype(int)
    cv2.circle(vis_map.data, (robot_rc_est[1], robot_rc_est[0]), 5, (255, 0, 0), -1)
    vis_map.data = cv2.arrowedLine(vis_map.data, (robot_rc_est[1], robot_rc_est[0]),
                                   (robot_rc_est[1] + np.round(20 * np.cos(robot_pose_est[2])).astype(np.int),
                                    robot_rc_est[0] - np.round(20 * np.sin(robot_pose_est[2])).astype(np.int)),
                                   (255, 0, 0), 2, tipLength=0.5)

    x_cov = state_covariance[:2, :2]
    w, e = np.linalg.eig(100 * x_cov)
    angle = np.arctan2(e[1, 0], e[0, 0]) * 180 / np.pi
    vis_map.data = cv2.ellipse(vis_map.data, (robot_rc_est[1], robot_rc_est[0]), (int(0.1+np.abs(w[1])), int(0.1+np.abs(w[0]))), angle, 0, 360,
                               (255, 0, 0), 3)

    for i in range(landmarks_gt.shape[0]):
        landmark_rc_gt = xy_to_rc(landmarks_gt[i], vis_map).astype(int)
        landmark_rc_est = xy_to_rc(landmarks_est[2 * i:2 * i + 2], vis_map).astype(int)
        cv2.circle(vis_map.data, (landmark_rc_gt[1], landmark_rc_gt[0]), 5, (0, 0, 255), -1)
        cv2.circle(vis_map.data, (landmark_rc_est[1], landmark_rc_est[0]), 5, (255, 255, 0), -1)

        landmark_cov = state_covariance[(3 + i * 2):(5 + i * 2), (3 + i * 2):(5 + i * 2)]
        w, e = np.linalg.eig(100 * landmark_cov)
        angle = np.arctan2(e[1, 0], e[0, 0]) * 180 / np.pi
        vis_map.data = cv2.ellipse(vis_map.data, (landmark_rc_est[1], landmark_rc_est[0]), (int(0.1+np.abs(w[1])), int(0.1+np.abs(w[0]))),
                                   angle, 0, 360, (255, 255, 0), 3)

    if trajectory is not None:
        for mu in trajectory:
            robot_position_rc = xy_to_rc(mu[:2], vis_map).astype(int)
            cv2.circle(vis_map.data, (robot_position_rc[1], robot_position_rc[0]), 2, (0, 255, 0), -1)

    if save_file is not None:
        assert isinstance(save_file, str)
        cv2.imwrite(save_file, vis_map.data)
        return
    else:
        cv2.namedWindow('map', cv2.WINDOW_GUI_NORMAL)
        cv2.imshow('map', vis_map.data)
        cv2.resizeWindow('map', *render_size)
        cv2.waitKey(1)