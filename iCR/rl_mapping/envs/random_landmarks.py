import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
from bc_exploration.envs.grid_world import GridWorld
from bc_exploration.utilities.util import rc_to_xy

from rl_mapping.sensors.position_sensor import PositionSensor
from rl_mapping.utilities.utils_landmarks import f


class RandomLandmarksEnv(GridWorld):
    def __init__(self, map_filename,
                 map_resolution,
                 sensor,
                 footprint,
                 motion_model_cov=np.diag(np.ones(3)),
                 tau=1,
                 num_landmarks=20,
                 start_state=None,
                 render_size=(500, 500)):

        assert isinstance(sensor, PositionSensor), "sensor model should be defined as \"PositionSensor\"."

        GridWorld.__init__(self,
                           map_filename=map_filename,
                           map_resolution=map_resolution,
                           sensor=sensor,
                           footprint=footprint,
                           start_state=start_state,
                           render_size=render_size)

        env_shape = self._map.get_shape()

        landmarks_rc = np.random.randint(low=(0, 0), high=env_shape, size=(num_landmarks, 2))
        self._landmarks_xy = rc_to_xy(landmarks_rc, self._map)

        self._sensor.set_landmarks(self._landmarks_xy)

        self._motion_model_cov = motion_model_cov
        self._tau = tau

    def step(self, action):
        noise = np.random.multivariate_normal(np.zeros(3), self._motion_model_cov)
        current_state = self._state.copy()
        desired_state = f(current_state, action, noise, self._tau)
        return GridWorld.step(self, desired_state=desired_state)

    def get_landmarks_position_gt(self):
        return self._landmarks_xy

    def get_noisy_state(self):
        robot_pose = self._state.copy()
        landmarks = self._landmarks_xy + 20 * np.random.multivariate_normal(np.zeros(2), self._motion_model_cov[:2, :2],
                                                                            size=self._landmarks_xy.shape[0])
        return robot_pose, landmarks
