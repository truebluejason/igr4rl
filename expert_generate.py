import argparse
import os
import pandas as pd
import torch

from expert_train import get_mt10_env, set_seed
from igr4rl import evaluate_on_env
from soft_actor_critic import SAC
from utils import DotDict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to saved train run info')
    parser.add_argument('--render', action="store_true",
                        help='Render evaluation episodes (default: False)')
    parser.add_argument('--n_samples', default=1000000, type=int,
                        help='Number of state/action pairs to sample from the expert')
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path)
    args = DotDict(checkpoint['train_configs'])
    args.cuda = False
    env_name, env, tasks = get_mt10_env(args.env_index, args.n_task)
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_checkpoint(path, evaluate=True)
    set_seed(args.seed, env)
    return agent, (env, env_name, tasks), args


def collect_trajectories(agent, env_info, n_samples, render):
    env, _, tasks = env_info
    trajectories, avg_rewards = [], []
    collected = 0
    while collected < n_samples:
        avg_reward, state_action_info = evaluate_on_env(agent, env, tasks, n_episodes=50,
                                                        random=False, collect_data=True, render=render)
        states, actions = state_action_info['state'], state_action_info['action']
        state_df = pd.DataFrame(states).rename(mapper=lambda n: f's_{n}', axis=1)
        action_df = pd.DataFrame(actions).rename(mapper=lambda n: f'a_{n}', axis=1)
        trajectories.append(pd.concat((state_df, action_df), axis=1))
        avg_rewards.append(avg_reward)
        collected += len(state_df)
        if collected % 1e5 == 0:
            print(f"Collected {collected} samples...")
    avg_reward = sum(avg_rewards)/len(avg_rewards)
    return pd.concat(trajectories, axis=0)[:n_samples], avg_reward


def save_result(env_name, data, avg_reward, seed):
    data_dir = os.path.join('data', env_name)
    result_dir = os.path.join('result', env_name)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    data_path = os.path.join(data_dir, 'expert.csv')
    result_path = os.path.join(result_dir, f'{env_name}.csv')
    data.to_csv(data_path, index=False)
    if os.path.exists(result_path):
        result_df = pd.read_csv(result_path)
    else:
        result_df = pd.DataFrame({'agent': [], 'seed': [], 'avg_reward': []})
    new_row = {'agent': 'expert', 'seed': seed, 'avg_reward': avg_reward}
    result_df = result_df.append(new_row, ignore_index=True)
    result_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    agent, env_info, train_args = load_checkpoint(args.checkpoint)
    print(f"Loaded train run with arguments:\n{train_args}")
    data, avg_reward = collect_trajectories(agent, env_info, args.n_samples, args.render)
    save_result(env_info[1], data, avg_reward, train_args.seed)
