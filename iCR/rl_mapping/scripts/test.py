from stable_baselines3 import DDPG, PPO

import os, sys
import yaml

cur_path = os.path.abspath(os.path.dirname(__file__))
print(cur_path)
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))
print(os.path.abspath("."))

from rl_mapping.envs.env import VolumetricQuadrotor

NUM_TEST = 1

def test(params_filepath: str, ckpt_name: str=""):
    ### read parameters ###
    with open(os.path.join(os.path.abspath("."), params_filepath)) as f:
            params = yaml.load(f, Loader=yaml.FullLoader)

    ### create env ###
    env_params_filepath = os.path.join(os.path.split(params_filepath)[0], os.path.split(params['env_params_filepath'])[1])
    env = VolumetricQuadrotor(params['map_filepath'], env_params_filepath)

    ### load model ###
    if ckpt_name == "":
        ckpt_name = params['exp_name']
    if params['algorithm'] == 'PPO':
        agent = PPO.load(os.path.join(params['checkpoints_folder'], params['exp_name'], ckpt_name), device='cpu')
    elif params['algorithm'] == 'DDPG':
        agent = DDPG.load(os.path.join(params['checkpoints_folder'], params['exp_name'], ckpt_name), device='cpu')
    else:
        raise NotImplementedError

    ### get parameters ###
    try:
        gamma = float(agent.gamma)
    except:
        gamma = 1.
    print(f"gamma = {gamma}")

    ### test ###
    for eps in range(NUM_TEST):

        obs = env.reset()
        done = False
        total_reward = 0

        print(f"\n------ Eps {eps} ------")
        print(f"init pose = {obs['pose']}")

        while not done:
            # get action
            action, _state = agent.predict(obs)
            print("action =", action)

            # step env
            obs, r, done, info = env.step(action)
            print("obs =", obs['pose'])

            # calc return
            total_reward += r * (gamma ** env.current_step)

            # render
            env.render(mode='human')
            print(f"reward = {r}\n")

        # summary
        print("---")
        print(f"return = {total_reward}\n")
    env.close()

if __name__ == '__main__':
    # test("checkpoints/ppo/no_bound/training_params.yaml")
    test("checkpoints/ppo/no_bound/training_params.yaml", "no_bound_800000_steps.zip")

        
