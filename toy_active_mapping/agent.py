import numpy as np
from env import SimpleQuadrotor
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

NUM_STEPS = 1e5
LOG_INTERVAL=50

def make_ddpg_agent(env):
    # action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # model
    # model = DDPG('MlpPolicy', env, action_noise=action_noise, buffer_size=200000, learning_starts=10000, gamma=0.98, policy_kwargs=dict(net_arch=[400, 300]), verbose=1, tensorboard_log="./tensorboard/ddpg_toy_active_mapping/") # zoo
    model = DDPG('MlpPolicy', env, action_noise=action_noise, verbose=1, tensorboard_log="./toy_active_mapping/tensorboard/ddpg_toy_active_mapping/") # default

    return model

if __name__ == '__main__':
    # init env
    env = SimpleQuadrotor()

    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=1)

    # train agent
    model = make_ddpg_agent(env)
    model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL, tb_log_name="default")
    model.save("./toy_active_mapping/checkpoints/ddpg_toy_active_mapping/default")