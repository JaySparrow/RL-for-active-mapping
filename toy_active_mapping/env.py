import math
import numpy as np
from numpy.linalg import slogdet

import gym
from gym import spaces

from stable_baselines3.common.env_checker import check_env

import matplotlib.pyplot as plt

from utils import unicycle_dyn, diff_FoV_land

# env setting
STATE_DIM = 3
LANDMARK_DIM = 2
LANDMARK_NUM = 2
RADIUS = .2
STD = 0.5
KAPPA = 0.2

# time & step
TOTAL_TIME = 5
STEP_SIZE = 1

class SimpleQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self):
        super(SimpleQuadrotor, self).__init__()

        # variables
        self.total_time = TOTAL_TIME
        self.step_size = STEP_SIZE
        self.total_step = math.floor(TOTAL_TIME / STEP_SIZE)

        # action space
        # defined as {-1, 1} as suggested by stable_baslines3, rescaled to {-2, 2} later in step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(STATE_DIM, ), dtype=np.float32) # (x, y, \theta): {-2, 2}^3
        
        # state space
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM, ), dtype=np.float32) # (x, y, \theta): {-inf, inf}^3

        # state init
        self.state = np.zeros(STATE_DIM, dtype=np.float32)

        # landmark init
        self.landmark = np.array([[1], [1], [-1], [-1]], dtype=np.float32)
        self.info_mat = np.diag([1, 1, 2, 2]).astype(np.float32)

        # step counter init
        self.current_step = -1

        # plot
        self.history_states = [self.state]
        self.fig = None
        self.ax = None

    def step(self, action):
        self.current_step += 1

        # rescale actions
        action *= 2

        # dynamics
        next_state = unicycle_dyn(self.state, action, self.step_size).astype(np.float32)

        # reward
        V_jj_inv = diff_FoV_land(next_state, self.landmark, LANDMARK_NUM, RADIUS, KAPPA, STD).astype(np.float32) # diff_FoV
        next_info_mat = self.info_mat + V_jj_inv # update info
        reward = float(slogdet(next_info_mat)[1] - slogdet(self.info_mat)[1])

        # terminate at time
        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {'info_mat': next_info_mat}

        # update variables
        self.state = next_state
        self.info_mat = next_info_mat

        # record history poses
        self.history_states.append(self.state)

        return next_state, reward, done, info

    def reset(self):
        # state init
        self.state = np.zeros(STATE_DIM, dtype=np.float32)

        # landmark init
        self.landmark = np.array([[1], [1], [-1], [-1]], dtype=np.float32)
        self.info_mat = np.diag([1, 1, 2, 2]).astype(np.float32)

        # step counter init
        self.current_step = -1

        # plot
        self.history_states = [self.state]
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        return self.state

    def _plot(self):

        # plot agent trajectory
        history_states = np.array(self.history_states)
        self.ax.plot(history_states[:, 0], history_states[:, 1], c='black', label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_states[0, 0], history_states[0, 1], marker='>', s=10, c='red', label="start")
        self.ax.scatter(history_states[-1, 0], history_states[-1, 1], marker='s', s=10, c='red', label="end")

        # plot landmarks
        self.ax.scatter(self.landmark[[0, 2], :], self.landmark[[1, 3], :], s=10, c='blue', label="landmark")

        # axes
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        # legend
        self.ax.legend()

    def render(self, mode='human'):
        if mode == 'terminal':
            print(f">>>> step {self.current_step} <<<<")
            print(f"state = {self.state}")
            print("information matrix:")
            print(self.info_mat)
        elif mode == 'human':
            # clear axes
            self.ax.cla()

            # plot
            self._plot()

            # display
            plt.draw()
            plt.pause(0.5)

        else:
            raise NotImplementedError

    def save_plot(self, name='default.png'):
        self.ax.cla()
        self._plot()
        self.fig.savefig(name)

    def close (self):
        plt.close('all')

if __name__ == '__main__':
    # create env
    env = SimpleQuadrotor()
    check_env(env)

    # # testing actions
    # action_space = np.array([np.pi/2*(np.arange(5)/5-0.5), 0.5*(np.arange(5)/5-0.5), np.zeros(5)], dtype=np.float32)
    # print(action_space)

    for eps in range(2):
        obs = env.reset()
        done = False
        print(f"\n------ Eps {eps} ------")
        print(f"init state = {obs}")
        while not done:
            action = env.action_space.sample() # action_space[:, env.current_step+1]
            obs, r, done, info = env.step(action)
            env.render(mode='human')
            print(f"reward = {r}")
        env.save_plot(name=f'plots/eps{eps}.png')
    env.close()
    
