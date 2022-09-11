import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), "../../"))

from rl_mapping.agents.icr_agent import create_icr_landmarks_agent_from_params
from rl_mapping.envs.env_test import SimpleQuadrotor

parser = argparse.ArgumentParser(description='landmark-based mapping')
parser.add_argument('--num-landmarks', type=int, default=5)
parser.add_argument('--horizon', type=int, default=15)
parser.add_argument('--bound', type=int, default=10)
parser.add_argument('--learning-curve-path', default="tensorboard/ppo_toy_active_mapping/")
parser.add_argument('--model-path', default="checkpoints/ppo_toy_active_mapping/default")
args = parser.parse_args()
NUM_TEST = 20
icr_landmarks_agent = create_icr_landmarks_agent_from_params(
    params_filename=os.path.join(os.path.abspath(os.path.join("", os.pardir)), "params/params_landmarks.yaml"), tau=1,
                                                      num_landmarks=args.num_landmarks, position_cov=0.5 * np.diag(np.array([1, 1])))

def test_agent():
    landmarks = np.random.uniform(low=-args.bound, high=args.bound, size=(args.num_landmarks * 2, 1))
    env = SimpleQuadrotor(args.num_landmarks, args.horizon, landmarks, True)
    for i in range(NUM_TEST):
        obs = env.reset()
        env.render(mode='human')
        total_reward = 0

        u = icr_landmarks_agent.plan(np.append(obs[:2], 0), mu_landmarks=obs[2 + args.num_landmarks * 2:],
                                 sigma_landmarks=np.linalg.inv(np.diag(obs[2: 2 + args.num_landmarks * 2])), mode='icr')

        print("u:", np.append(obs[:2], 0), obs[2 + args.num_landmarks * 2:],
              np.linalg.inv(np.diag(obs[2: 2 + args.num_landmarks * 2])), u, obs, "\n\n\n")

        for k in range(args.horizon):
            obs, r, done, info = env.step(u[:, k])
            total_reward += r
            env.render(mode='human')
            print("action:", u[:, k])

        print(total_reward)

if __name__ == '__main__':
    test_agent()