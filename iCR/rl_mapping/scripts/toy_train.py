import numpy as np
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

import os, sys, shutil
import yaml

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from rl_mapping.envs.toy_env import ToyQuadrotor
from rl_mapping.networks.toy_policy import CustomCNN

def train(params_filepath: str):
    ### read parameters ###
    with open(os.path.join(os.path.abspath("."), params_filepath)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    ### create env ###
    env = ToyQuadrotor(params['map_filepath'], params['env_params_filepath'])
    # wrap with vector env
    env = make_vec_env(lambda: env, n_envs=params['num_envs'])

    ### create agent model ###
    # policy network
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128)
    )
    # model
    if params['algorithm'] == 'DDPG':
        # action noise
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        model = DDPG('CnnPolicy', env, action_noise=action_noise, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=params['tensorboard_folder'], gamma=params['gamma'], learning_rate=params['lr'], seed=params['seed'], **params['ddpg'])
    elif params['algorithm'] == 'PPO':
        model = PPO('CnnPolicy', env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=params['tensorboard_folder'], gamma=params['gamma'], learning_rate=params['lr'], seed=params['seed'], **params['ppo'])
    else:
        raise NotImplementedError

    ### checkpoint ###
    # create checkpoint path/folder
    ckpt_path = os.path.join(params['checkpoints_folder'], params['exp_name'])
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    # copy and save parameter files
    dst = shutil.copy(params_filepath, ckpt_path)
    print("File copied to", dst)
    dst = shutil.copy(params['env_params_filepath'], ckpt_path)
    print("File copied to", dst)
    # define checkpoint save behaviour
    checkpoint_callback = CheckpointCallback(save_freq=params['save_frequency'], save_path=ckpt_path, name_prefix=params['exp_name'])

    ### evaluation ###
    # eval env
    eval_env = ToyQuadrotor(params['map_filepath'], params['env_params_filepath'])
    # eval callback
    eval_callback = EvalCallback(eval_env, best_model_save_path=ckpt_path, log_path=ckpt_path, eval_freq=int(params['save_frequency']/5), deterministic=True, render=False)

    ### train agent ###
    model.learn(total_timesteps=params['num_steps'], log_interval=params['log_interval'], tb_log_name=params['exp_name'], callback=[eval_callback, checkpoint_callback])
    model.save(os.path.join(ckpt_path, params['exp_name']))

if __name__ == '__main__':
    train("../params/toy_training_params.yaml")