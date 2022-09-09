import numpy as np
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from rl_mapping.utilities.utils_landmarks import state_to_T


class PositionSensor:
    def __init__(self, sensor_range, angular_range, position_cov):
        self._sensor_range = sensor_range
        self._angular_range = angular_range
        self._position_cov = position_cov

        self._landmarks = None
        self._landmarks_h = None

    def set_landmarks(self, landmarks):
        self._landmarks = landmarks
        self._landmarks_h = np.hstack((self._landmarks, np.ones((landmarks.shape[0], 1))))

    def set_map(self, occupancy_map):
        return

    def measure(self, state):
        if self._landmarks is None:
            return None
        T_robot = state_to_T(state)
        landmarks_robot_h = self._landmarks_h @ np.transpose(np.linalg.inv(T_robot))
        landmarks_robot = landmarks_robot_h[:, :2]
        landmarks_robot_noisy = self._add_noise(landmarks_robot)
        ranges = np.linalg.norm(landmarks_robot, axis=1)
        angles = np.arctan2(landmarks_robot[:, 1], landmarks_robot[:, 0])
        visible_landmarks = (0 <= ranges) * (ranges <= self._sensor_range) *\
                            (- self._angular_range / 2 <= angles) * (angles <= self._angular_range / 2)
        landmarks_robot_noisy[np.logical_not(visible_landmarks), :] = 0
        return landmarks_robot_noisy

    def _add_noise(self, landmarks):
        noise = np.random.multivariate_normal(mean=np.zeros(2), cov=self._position_cov, size=landmarks.shape[0])
        return landmarks + noise
