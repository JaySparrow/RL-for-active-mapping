import os
import sys
import yaml
import cv2
import numpy as np
import time

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from bc_exploration.mapping.costmap import Costmap
from bc_exploration.utilities.util import rc_to_xy, xy_to_rc
from bc_exploration.footprints.footprints import CustomFootprint
from bc_exploration.footprints.footprint_points import get_tricky_circular_footprint, get_tricky_oval_footprint

from rl_mapping.utilities.utils import state_to_T, T_to_state, SE2_motion, Gaussian_CDF, circle_SDF, visualize
from rl_mapping.sensors.semantic_sensors import SemanticLidar
from rl_mapping.envs.semantic_grid_world import SemanticGridWorld
from rl_mapping.mapping.mapper_kf import KFMapper

class ToyQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    ACTIONS = {
        0: np.array([0., 0.]), # no move
        1: np.array([1., 0.]), # E
        2: np.array([1/np.sqrt(2), 1/np.sqrt(2)]), # NE
        3: np.array([0., 1.]), # N
        4: np.array([-1/np.sqrt(2), 1/np.sqrt(2)]), # NW
        5: np.array([-1., 0.]), # W
        6: np.array([-1/np.sqrt(2), -1/np.sqrt(2)]), # SW
        7: np.array([0., -1.]), # S
        8: np.array([1/np.sqrt(2), -1/np.sqrt(2)]) # SE
    }

    def __init__(self, map_filename: str, params_filename: str):
        super(ToyQuadrotor, self).__init__()

        ## parameters
        self.__load_params(params_filename)

        ## distribution map
        self.__load_distrib_map(map_filename)

        h, w, _ = self.distrib_map.get_shape()
        self.xmax, self.ymax, _ = self.distrib_map.get_size()
        self.ox, self.oy = self.distrib_map.origin

        ## observation space: binary
        # layer 1: detection map (1: detected; 0: undetected)
        # layer 2: agent position (1: agent; 0: no agent)
        # self.observation_space = spaces.MultiBinary([2, h, w])
        self.observation_space = spaces.Box(low=0, high=1, shape=(2, h, w), dtype=int)

        ## action space: discrete  
        # velocities in the 8 directions with unit magnitude (i.e. \sqrt(vx^2 + vy^2) = 1) & no move
        self.action_space = spaces.Discrete(9)

        ## init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.detection_map = np.zeros(self.info_vec.shape, dtype=int) # (h, w)

        # corner
        # self.agent_pos = np.zeros(3, dtype=np.float32)
        # center
        self.agent_pos = np.array([self.ox + self.xmax/2, self.oy + self.ymax/2, 0.], dtype=np.float32) # (3, )

        self.current_step = -1

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}!"
        self.current_step += 1

        ## rescale actions
        control = self.control_scale * np.hstack([
            self.ACTIONS[action], 
            0
        ]).astype(np.float32)

        ## apply dynamics and update agent's pos
        T_old = state_to_T(self.agent_pos)
        T_new = SE2_motion(T_old, control, self.dt)
        self.agent_pos = T_to_state(T_new)

        ## boundary enforce
        bounded_pos = np.clip(self.agent_pos[:2], [self.ox, self.oy], [self.xmax + self.ox, self.ymax + self.oy])
        self.agent_pos[:2] = bounded_pos

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

        ## reward = percentage of newly detected cells \in [0, 1]
        h, w = self.info_vec.shape
        new_cells = np.logical_and(np.logical_not(np.isclose(Phi, 1.)), np.logical_not(self.detection_map))
        r = float(np.sum(new_cells)) / (h * w)

        ## update detection map
        self.detection_map[np.where(new_cells)] = 1

        # obs
        agent_r, agent_c = np.clip(xy_to_rc(self.agent_pos, self.distrib_map).astype(int)[:2], [0, 0], [h-1, w-1])
        agent_pos_map = np.zeros(self.info_vec.shape, dtype=int) # (h, w)
        agent_pos_map[agent_r, agent_c] = 1
        obs = np.stack([self.detection_map, agent_pos_map]).astype(int)

        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {}

        return obs, r, done, info

    def reset(self):

        # init
        self.info_vec = self.distrib_map.data[:, :, 1].copy() # (h, w)
        self.detection_map = np.zeros(self.info_vec.shape, dtype=int) # (h, w)
        
        # corner
        # self.agent_pos = np.zeros(3, dtype=np.float32)
        # center
        self.agent_pos = np.array([self.ox + self.xmax/2, self.oy + self.ymax/2, 0.], dtype=np.float32) # (3, )

        self.current_step = -1

        # construct observation
        agent_r, agent_c, _ = xy_to_rc(self.agent_pos, self.distrib_map).astype(int)
        agent_pos_map = np.zeros(self.info_vec.shape, dtype=int) # (h, w)
        agent_pos_map[agent_r, agent_c] = 1

        obs = np.stack([self.detection_map, agent_pos_map]).astype(int)

        return obs

    def render(self, mode='human'):
        # get obs
        pose, obs = self.__grid_env.step(self.agent_pos)
        # get occupancy map
        occ_map = self.mapper.update(state=pose, obs=obs)
        # visulization
        visualize(state=pose, semantic_map=occ_map, num_class=1, render_size=(np.array(self.__grid_env.get_map_shape()[::-1])).astype(int), wait_key=0, save_file=None)
        time.sleep(0.2)

    def close (self):
        cv2.destroyAllWindows()

    def __load_params(self, params_filename: str):
        with open(os.path.join(os.path.abspath("."), params_filename)) as f:
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
        self.__grid_env = SemanticGridWorld(map_filename=map_filename,
                                map_resolution=0.03,
                                sensor=sensor,
                                num_class=1,
                                footprint=footprint,
                                start_state=[0., 0., 0.],
                                no_collision=True)
        padding = 0.
        map_shape = np.array(self.__grid_env.get_map_shape()) + int(2. * padding // 0.03)
        initial_map = Costmap(data=Costmap.UNEXPLORED * np.ones(map_shape, dtype=np.uint8),
                            resolution=self.__grid_env.get_map_resolution(),
                            origin=[-padding - self.__grid_env.start_state[0], -padding - self.__grid_env.start_state[1]])

        self.mapper = KFMapper(initial_map=initial_map, sigma2=self.std**2)

        self.distrib_map = self.mapper.get_distrib_map()

if __name__ == '__main__':
    params_filename = '../params/toy_env_params.yaml'
    map_filename = '../maps/map6_converted.png'

    ### create env & test random actions ###
    env = ToyQuadrotor(map_filename, params_filename)
    check_env(env)
    print("env size    :", env.xmax, env.ymax)
    print("starting pos:", env.agent_pos[:-1])

    obs = env.reset()
    done = False
    total_reward = 0

    actions = [1 for _ in range(5)] +\
              [3 for _ in range(8)] +\
              [5 for _ in range(10)] +\
              [7 for _ in range(7)]

    time.sleep(3)
    while not done:
        action = actions[env.current_step+1]
        print("action =", env.ACTIONS[action])

        obs, r, done, info = env.step(action)
        print("pos =", np.hstack(np.where(np.isclose(obs[1], 1))))

        total_reward += r

        print(f"reward = {r}\n")
        env.render()

    print("---")
    print("return:", total_reward)

    env.close()

