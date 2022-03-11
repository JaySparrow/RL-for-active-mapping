import numpy as np
from env import SimpleQuadrotor
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise

NUM_STEPS = 1e5
LOG_INTERVAL=1

def make_ppo_agent(env):
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./toy_active_mapping/tensorboard/ddpg_toy_active_mapping/") # default
    return model

if __name__ == '__main__':
    # init env
    env = SimpleQuadrotor()

    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=1)

    # train agent
    model = make_ppo_agent(env)
    model.learn(total_timesteps=NUM_STEPS, log_interval=LOG_INTERVAL, tb_log_name="default")
    model.save("./toy_active_mapping/checkpoints/ddpg_toy_active_mapping/default")