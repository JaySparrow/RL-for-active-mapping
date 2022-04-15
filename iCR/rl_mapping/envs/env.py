import os
import sys
import yaml
import numpy as np

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import rc_to_xy
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint

from rl_mapping.utilities.utils import state_to_T, T_to_state, SE2_motion, Gaussian_CDF, circle_SDF
from rl_mapping.sensors.semantic_sensors import SemanticLidar
from rl_mapping.envs.semantic_grid_world import SemanticGridWorld
from rl_mapping.mapping.mapper_kf import KFMapper

class VolumetricQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self, map_filename: str, params_filename: str):
        super(VolumetricQuadrotor, self).__init__()

        ## parameters
        self.__load_params(params_filename)

        ## distribution map
        self.__load_distrib_map(map_filename)

        ## observation space: diagonal entries of the information matrix (i.e. the information vector) + agent position
        h, w, _ = self.distrib_map.get_shape()
        self.xmax, self.ymax, _ = self.distrib_map.get_size()
        self.ox, self.oy = self.distrib_map.origin
        self.observation_space = spaces.Dict({
            'info': spaces.Box(low=-np.inf, high=np.inf, shape=(1, h, w), dtype=np.float32),
            'pose': spaces.Box(low=np.array([self.ox, self.oy]), high=np.array([self.xmax + self.ox, self.ymax + self.oy]), dtype=np.float32)
        })

        ## action space: linearly controlling the x, y velocities, yaw angle is fixed as 0
        # (x, y): {-control_scale, control_scale}^2
        # scaled in the step function
        self.action_space = spaces.Box(low=-1, high=1, shape=(2, ), dtype=np.float32) 

        ## init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.agent_pos = np.array([self.ox + self.xmax/2, self.oy + self.ymax/2, 0.], dtype=np.float32) # (3, )
        self.last_r = np.sum(np.log(self.info_vec[::self.downsample_rate, ::self.downsample_rate]))
        self.current_step = -1

    def step(self, action):
        self.current_step += 1

        # rescale actions
        action *= self.control_scale
        control = np.hstack([
            action, 
            0
        ]).astype(np.float32)

        # apply dynamics and update agent's pos
        T_old = state_to_T(self.agent_pos)
        T_new = SE2_motion(T_old, control, self.dt)
        self.agent_pos = T_to_state(T_new)

        # boundary enforce
        self.agent_pos[:2] = np.clip(self.agent_pos[:2], [self.ox, self.oy], [self.xmax + self.ox, self.ymax + self.oy])

        ## update information: vectorized
        # downsample indices
        idx_mat = np.indices(self.info_vec.shape)[:, ::self.downsample_rate, ::self.downsample_rate] # (2, h / downsample_rate, w / downsample_rate) = (2, h', w')
        idx     = idx_mat.reshape((2, -1)).T # (h' x w', 2) = (N, 2)
        # convert row, column to x, y
        p_ij = rc_to_xy(idx, self.distrib_map) # (N, 2)
        # operate poses
        q = T_old[:2, :2].transpose() @ (p_ij - T_old[:2, 2]).T # (2, N)
        # SDF
        d, _ = circle_SDF(q, self.sensor_range) # (N, )
        # gaussian
        Phi, _ = Gaussian_CDF(d, self.kappa) # (N, )
        Phi = Phi.reshape((idx_mat.shape[1:])) # (h', w')
        # update information
        self.info_vec[::self.downsample_rate, ::self.downsample_rate] += 1 / (self.std**2) * (1 - Phi)

        # calculate reward
        cur_r = np.sum(np.log(self.info_vec[::self.downsample_rate, ::self.downsample_rate]))
        r = cur_r - self.last_r
        self.last_r = cur_r

        # obs
        obs = {
            'info': self.info_vec.astype(np.float32)[np.newaxis, :, :],
            'pose': self.agent_pos[:2]
        }

        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {}

        return obs, r, done, info

    def reset(self):

        # init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.agent_pos = np.array([self.ox + self.xmax/2, self.oy + self.ymax/2, 0.], dtype=np.float32) # (3, )
        self.last_r = np.sum(np.log(self.info_vec[::self.downsample_rate, ::self.downsample_rate]))
        self.current_step = -1

        # construct observation
        obs = {
            'info': self.info_vec.astype(np.float32)[np.newaxis, :, :],
            'pose': self.agent_pos[:2]
        }

        return obs

    def render(self, mode='human'):
        pass

    def close (self):
        pass

    def __load_params(self, params_filename: str):
        with open(os.path.join(params_filename)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

        # information
        self.std = params['std']
        self.kappa = params['kappa']
        # sensor
        self.downsample_rate = params['downsample_rate']
        self.sensor_range = params['sensor_range']
        # time
        self.total_time = params['horizon']
        self.dt = params['dt']
        self.total_step = np.floor(self.total_time / self.dt)
        # control
        self.control_scale = params['control_scale']
        # footprint
        self.footprint_type = params['footprint']['type']
        try:
            self.footprint_radius = params['footprint']['radius']
        except:
            pass
        self.footprint_angular_resolution = params['footprint']['angular_resolution']
        self.footprint_inflation_scale = params['footprint']['inflation_scale']

    def __load_distrib_map(self, map_filename: str):

        if self.footprint_type == 'tricky_circle':
            footprint_points = get_tricky_circular_footprint()
        elif self.footprint_type == 'tricky_oval':
            footprint_points = get_tricky_oval_footprint()
        elif self.footprint_type == 'circle':
            rotation_angles = np.arange(0, 2 * np.pi, 4 * np.pi / 180)
            footprint_points = \
                self.footprint_radius * np.array([np.cos(rotation_angles), np.sin(rotation_angles)]).T
        elif self.footprint_type == 'pixel':
            footprint_points = np.array([[0., 0.]])
        else:
            footprint_points = None
            assert False and "footprint type specified not supported."

        footprint = CustomFootprint(footprint_points=footprint_points,
                                angular_resolution=self.footprint_angular_resolution,
                                inflation_scale=self.footprint_inflation_scale)
        sensor = SemanticLidar(sensor_range=self.sensor_range,
                            angular_range=np.pi*2,
                            angular_resolution=0.5 * np.pi / 180,
                            map_resolution=0.03,
                            num_classes=1,
                            aerial_view=True)
        env = SemanticGridWorld(map_filename=map_filename,
                                map_resolution=0.03,
                                sensor=sensor,
                                num_class=1,
                                footprint=footprint,
                                start_state=[0., 0., 0.],
                                no_collision=True)
        padding = 0.
        map_shape = np.array(env.get_map_shape()) + int(2. * padding // 0.03)
        initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                            resolution=env.get_map_resolution(),
                            origin=[-padding - env.start_state[0], -padding - env.start_state[1]])

        mapper = KFMapper(initial_map=initial_map, sigma2=self.std**2)

        self.distrib_map = mapper.get_distrib_map()


if __name__ == '__main__':
    params_filename = '../params/env_params.yaml'
    map_filename = '../maps/map6_converted.png'

    ### create env & test random actions ###
    env = VolumetricQuadrotor(map_filename, params_filename)
    check_env(env)

    obs = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = env.action_space.sample()

        obs, r, done, info = env.step(action)

        total_reward += r

        print("reward: ", r)

    print("return:", total_reward)

    env.close()

