import math
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

from numpy.linalg import slogdet
from gym import spaces
from stable_baselines3.common.env_checker import check_env
from utils import unicycle_dyn, diff_FoV_land

# env setting
STATE_DIM = 3
RADIUS = .5
STD = 0.5
KAPPA = 1  # TODO: increase

# time & step
TOTAL_TIME = 50
STEP_SIZE = 1

random.seed(100)
np.random.seed(100)

class SimpleQuadrotor(gym.Env):
    metadata = {'render.modes': ['human', 'terminal']}

    def __init__(self, num_landmarks, test=None):
        super(SimpleQuadrotor, self).__init__()

        # variables
        self.num_landmarks = num_landmarks
        self.total_time = TOTAL_TIME
        self.step_size = STEP_SIZE
        self.total_step = math.floor(TOTAL_TIME / STEP_SIZE)

        # action space
        # defined as {-1, 1} as suggested by stable_baslines3, rescaled to {-2, 2} later in step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(STATE_DIM, ), dtype=np.float32) # (x, y, \theta): {-2, 2}^3
        
        # state space
        # agent state + diag of info mat
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(STATE_DIM + self.num_landmarks*2, ), dtype=np.float32) # (x, y, \theta, info_mat_0, info_mat_1, info_mat_2, info_mat_3): {-inf, inf}^7

        # landmark and info_mat init
        if test == None:
            self.landmark = np.random.uniform(low=-10, high=10.0, size=(self.num_landmarks * 2, 1))
        else:
            self.landmark = np.array([[ 0.86809884],
       [-4.4326123 ],
       [-1.50964819],
       [ 6.89552265],
       [-9.90562288],
       [-7.56861758],
       [ 3.41498169],
       [ 6.5170551 ],
       [-7.26586821],
       [ 1.50186659],
       [ 7.82643909],
       [-5.81595756],
       [-6.29343561],
       [-7.83246219],
       [-5.60605015],
       [ 9.57247569],
       [ 6.23366298],
       [-6.56117975],
       [ 6.32449497],
       [-4.51852506],
       [-1.36591633],
       [ 8.80059639],
       [ 6.35298758],
       [-3.277761  ],
       [-6.49179093],
       [-2.54335907],
       [-9.88622985],
       [-4.95147293],
       [ 5.91325017],
       [-9.69490058]])
        self.info_mat_init = np.diag([.5] * self.num_landmarks * 2).astype(np.float32)
        print("landmarks' positions:", [self.landmark])

    def step(self, action):
        self.current_step += 1

        # rescale actions
        action *= 3

        # record history action
        self.history_actions.append(action.copy())

        # enforece the 3rd dim to zero
        action[-1] = 0.

        # dynamics
        next_agent_pos = unicycle_dyn(self.agent_pos, action, self.step_size).astype(np.float32)

        # reward
        V_jj_inv = diff_FoV_land(next_agent_pos, self.landmark, self.num_landmarks, RADIUS, KAPPA, STD).astype(np.float32) # diff_FoV
        next_info_mat = self.info_mat + V_jj_inv # update info
        reward = float(slogdet(next_info_mat)[1] - slogdet(self.info_mat)[1])

        # terminate at time
        done = False
        if self.current_step >= self.total_step-1:
            done = True

        # info
        info = {'info_mat': next_info_mat}

        # update variables
        self.agent_pos = next_agent_pos
        self.info_mat = next_info_mat

        # update state
        self.state = np.hstack([
            self.agent_pos,
            self.info_mat.diagonal()
        ]).astype(np.float32)

        # record history poses
        self.history_poses.append(self.agent_pos)

        return self.state, reward, done, info

    def reset(self):
        # landmark and info_mat init
        self.info_mat = self.info_mat_init

        # agent pose init
        # self.agent_pos = np.zeros(STATE_DIM, dtype=np.float32)
        self.agent_pos = np.array([random.uniform(-2, 2), random.uniform(-3, 3), 0])
        # self.agent_pos = np.array([-3, 3, 0])

        # state init
        self.state = np.hstack([
            self.agent_pos,
            self.info_mat.diagonal()
        ]).astype(np.float32)

        # step counter init
        self.current_step = -1

        # plot
        self.history_poses = [self.agent_pos]
        self.history_actions = []
        self.fig = plt.figure(1)
        self.ax = self.fig.gca()

        return self.state

    def _plot(self, title='trajectory'):

        # plot agent trajectory
        history_poses = np.array(self.history_poses)
        self.ax.plot(history_poses[:, 0], history_poses[:, 1], c='black', label='agent trajectory')

        # plot agent trajectory start & end
        self.ax.scatter(history_poses[0, 0], history_poses[0, 1], marker='>', s=15, c='red', label="start")
        self.ax.scatter(history_poses[-1, 0], history_poses[-1, 1], marker='s', s=15, c='red', label="end")

        # plot landmarks
        # self.ax.scatter(self.landmark[[0, 2], :], self.landmark[[1, 3], :], s=10, c='blue', label="landmark")
        self.ax.scatter(self.landmark[list(range(0, self.num_landmarks*2, 2)), :],
                        self.landmark[list(range(1, self.num_landmarks*2+1, 2)), :], s=10, c='blue', label="landmark")

        # annotate theta value to each position point
        for i in range(0, len(self.history_poses)-1):
            self.ax.annotate(round(self.history_actions[i][2], 4), history_poses[i, :2])

        # axes
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        # title
        self.ax.set_title(title)

        # legend
        self.ax.legend()

    def render(self, mode='human'):
        if mode == 'terminal':
            print(f">>>> step {self.current_step} <<<<")
            print(f"agent pos = {self.agent_pos}")
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

    def save_plot(self, name='default.png', title='trajectory'):
        self.ax.cla()
        self._plot(title=title)
        self.fig.savefig(name)

    def close (self):
        plt.close('all')

if __name__ == '__main__':
    num_eps = 4
    gamma = 0.98
    ## create env
    env = SimpleQuadrotor()
    check_env(env)

    ## testing actions
    # eps0: diagonal /
    actions0 = np.array([
        [-0.5, -0.5, 0.],
        [-0.5, -0.5, 0.],
        [1.0, 1.0, 0.],
        [0.5, 0.5, 0.],
        [0.5, 0.5, 0.]
    ], dtype=np.float32).T/2
    # eps1: no move
    actions1 = np.zeros((5, 3), dtype=np.float32).T/2
    # eps2: diagonal \
    actions2 = np.array([
        [-0.5, 0.5, 0.],
        [-0.5, 0.5, 0.],
        [1.0, -1.0, 0.],
        [0.5, -0.5, 0.],
        [0.5, -0.5, 0.]
    ], dtype=np.float32).T/2
    action_spaces = [actions0, actions1, actions2]

    ## run examples
    for eps in range(num_eps):
        obs = env.reset()
        done = False
        total_reward = 0

        print(f"\n------ Eps {eps} ------")
        print(f"init state = {obs}")

        while not done:
            # get action
            if eps < len(action_spaces):
                action = action_spaces[eps][:, env.current_step+1]
            else:
                action = env.action_space.sample()

            # step env
            obs, r, done, info = env.step(action)

            # calc return
            total_reward += r * (gamma ** env.current_step)

            # render
            env.render(mode='human')
            print(f"reward = {r}")

        # summary
        print("---")
        print(env.history_actions)
        print(f"return = {total_reward}")
        env.save_plot(name=f'plots/eps{eps}.png', title=f'return = {total_reward}')
    env.close()
