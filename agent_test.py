import argparse
import os
import pandas as pd
import torch

from expert_train import get_mt10_env
from agent_train import get_env_info
from igr4rl import BehaviorCloningAgent, evaluate_on_env
from utils import DotDict


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True,
                        help='Path to saved train run info')
    parser.add_argument('--render', action="store_true",
                        help='Render evaluation episodes (default: False)')
    parser.add_argument('--n_episodes', default=50, type=int,
                        help='Number of episodes to evaluate the agent against')
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path)
    args = DotDict(checkpoint['train_configs'])
    sample_env = get_mt10_env(0, 1)[1]
    input_dim, action_space = sample_env.observation_space.shape[0], sample_env.action_space
    agent = BehaviorCloningAgent(input_dim, action_space, args.num_layers, args.hidden_dim)
    agent.load_checkpoint(path, evaluate=True)
    env_info = get_env_info(args.env_indexes, args.n_task, args.seed)
    return agent, env_info, args


def evaluate_envs(agent, env_info, n_episodes, render):
    # log overall and per environment performance
    task_avg_rewards = {}
    for info in env_info.values():
        env_name, env, tasks = info
        avg_reward, _ = evaluate_on_env(agent, env, tasks, n_episodes,
                                        random=False, collect_data=False, render=render)
        task_avg_rewards[env_name] = (avg_reward)
    task_avg_rewards['eval_return'] = sum(list(task_avg_rewards.values()))/len(task_avg_rewards)
    return task_avg_rewards


def save_result(run_name, env_info, task_avg_rewards, seed):
    for info in env_info.values():
        env_name = info[0]
        result_dir = os.path.join('result', env_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_path = os.path.join(result_dir, f'{env_name}.csv')
        if os.path.exists(result_path):
            result_df = pd.read_csv(result_path)
        else:
            result_df = pd.DataFrame({'agent': [], 'seed': [], 'avg_reward': []})
        new_row = {'agent': run_name, 'seed': seed, 'avg_reward': task_avg_rewards[env_name]}
        result_df = result_df.append(new_row, ignore_index=True)
        result_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    args = parse_arguments()
    agent, env_info, train_args = load_checkpoint(args.checkpoint)
    agent.to(torch.device('cpu'))
    print(f"Loaded train run with arguments:\n{train_args}")
    task_avg_rewards = evaluate_envs(agent, env_info, args.n_episodes, args.render)
    save_result(train_args.run_name, env_info, task_avg_rewards, train_args.seed)
