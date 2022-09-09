import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))

from rl_mapping.agents.icr_landmarks_agent import create_icr_landmarks_agent_from_params
from rl_mapping.estimation.ekf import EKF
from rl_mapping.utilities.viz_utils import visualize
from rl_mapping.envs.random_landmarks import RandomLandmarksEnv
from rl_mapping.sensors.position_sensor import PositionSensor


def run_icr_agent(map_filename, params_filename, sensor_range, angular_range,
                  position_cov, motion_cov, num_landmarks, tau, map_resolution,
                  start_state=None, render=True, render_interval=1,
                  render_size_scale=1.7, max_exploration_iterations=None):

    # define agent
    icr_landmarks_agent = create_icr_landmarks_agent_from_params(params_filename=params_filename, sensor_range=sensor_range,
                                                      angular_range=angular_range, tau=tau,
                                                      num_landmarks=num_landmarks,
                                                      motion_model_cov=motion_cov, position_cov=position_cov)
    footprint = icr_landmarks_agent.get_footprint()

    # pick a sensor
    sensor = PositionSensor(sensor_range=sensor_range,
                            angular_range=angular_range,
                            position_cov=position_cov)

    # setup grid world environment
    env = RandomLandmarksEnv(map_filename=map_filename,
                             map_resolution=map_resolution,
                             sensor=sensor,
                             footprint=footprint,
                             motion_model_cov=motion_cov,
                             tau=tau,
                             num_landmarks=num_landmarks,
                             start_state=start_state)

    render_size = (np.array(env.get_map_shape()[::-1]) * render_size_scale).astype(np.int)

    landmarks_gt = env.get_landmarks_position_gt()
    robot_pose, landmarks = env.get_noisy_state()

    ekf = EKF(num_landmarks=num_landmarks, motion_model_cov=motion_cov, position_cov=position_cov, tau=tau,
              init_robot_pose=robot_pose, init_landmark_pose=landmarks)

    # reset the environment to the start state, map the first observations
    pose_gt, obs = env.reset()

    mu, sigma = ekf.update(z=obs)

    pose_gt_list = [pose_gt]
    est_mu_list = [mu]
    est_sigma_list = [sigma]

    if render:
        # visualize(occupancy_map=semantic_map, state=pose, footprint=footprint,
        #           start_state=start_state, scan_angles=scan_angles, scan_ranges=scan_ranges,
        #           render_size=render_size, wait_key=0 if render_wait_for_key else 1)
        visualize(pose_gt, mu[:3], landmarks_gt, mu[3:], sigma,
                  env.get_map_shape(), env.get_map_resolution(), env.get_map_origin(), render_size, pose_gt_list,
                  save_file=None)

    iteration = 0
    is_last_plan = False
    plan_exists = False
    while True:

        if max_exploration_iterations is not None and iteration > max_exploration_iterations:
            is_last_plan = True

        # using the current map, make an action plan
        if plan_exists:
            u_opt = u[:, k]
            k += 1
            if k == u.shape[1]:
                plan_exists = False
                k = 0
        else:
            u = icr_landmarks_agent.plan(robot_pose=mu[:3], mu_landmarks=mu[3:], sigma_landmarks=sigma[3:, 3:], mode='icr')
            u_opt = u[:, 0]
            plan_exists = True
            k = 1

        pose_gt, obs = env.step(u_opt)
        ekf.predict(u_opt)
        mu, sigma = ekf.update(z=obs)

        pose_gt_list.append(pose_gt)
        est_mu_list.append(mu)
        est_sigma_list.append(sigma)

        # shows a live visualization of the exploration process if render is set to true
        if render:
            visualize(pose_gt, mu[:3], landmarks_gt, mu[3:], sigma,
                      env.get_map_shape(), env.get_map_resolution(), env.get_map_origin(), render_size, pose_gt_list,
                      save_file=None)

        if is_last_plan:
            visualize(pose_gt, mu[:3], landmarks_gt, mu[3:], sigma,
                      env.get_map_shape(), env.get_map_resolution(), env.get_map_origin(), render_size, pose_gt_list,
                      save_file='viz_icr.png')
            break

        iteration += 1

    if render:
        cv2.waitKey(0)

    # est_mu_list = np.array(est_mu_list)
    # est_sigma_list = np.array(est_sigma_list)
    # pose_gt_list = np.array(pose_gt_list)

    return est_mu_list, est_sigma_list, pose_gt_list, landmarks_gt


def main():
    """
    Main Function
    """
    np.random.seed(1)
    motion_cov = 0.0005 * np.diag(np.array([1, 1, 0.1]))
    position_cov = 0.5 * np.diag(np.array([1, 1]))
    num_landmarks = 5
    run_icr_agent(map_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "maps/blank_large.png"),
                  params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params_landmarks.yaml"),
                  map_resolution=0.1,
                  start_state=np.array([30, 30, 0]),
                  sensor_range=20.0,
                  angular_range=2*np.pi/3,
                  max_exploration_iterations=800,
                  render=True,
                  motion_cov=motion_cov,
                  position_cov=position_cov,
                  num_landmarks=num_landmarks,
                  tau=0.5,
                  render_size_scale=1)

    # print("Map", "{:.2f}".format(percent_explored * 100), "\b% explored!",
    #       "This is " + str(iterations_taken + 1) + " iterations!")


if __name__ == '__main__':
    main()