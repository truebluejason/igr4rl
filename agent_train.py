import argparse
import pandas as pd
import torch
import wandb

from expert_train import get_mt10_env, set_seed
from igr4rl import BehaviorCloningAgent, Trainer
from utils import DotDict

# os.environ["WANDB_MODE"] = "offline"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', required=True,
                        help='name of the run')
    parser.add_argument('--datasets', nargs='+', required=True,
                        help='Path to expert trajectory CSV')
    parser.add_argument('--env-indexes', nargs='+', required=True, type=int,
                        help='Indexes of the MT10 environment to train on (must be in same order as datasets)')
    parser.add_argument('--switch-env-every', default=1, type=int,
                        help='How many steps to take before switching environment')
    parser.add_argument('--n-task', default=10, type=int,
                        help='Number of tasks to run on for per environment')
    parser.add_argument('--render', action="store_true",
                        help='Render evaluation episodes (default: False)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_layers', type=int, default=3, metavar='N',
                        help='number of hidden layers')
    parser.add_argument('--hidden_dim', type=int, default=256, metavar='N',
                        help='hidden layer dimension')
    parser.add_argument('--num_steps', type=int, default=1000000, metavar='N',
                        help='maximum number of steps')
    parser.add_argument('--max_samples', type=int, default=5000000, metavar='N',
                        help='max number of samples to take from dataset')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    return parser.parse_args()


def get_env_info(env_indexes, n_task, seed):
    # create environments and tasks
    env_info = {}
    for env_index in env_indexes:
        env_name, env, task = get_mt10_env(env_index, n_task)
        set_seed(seed, env)
        env_info[env_index] = (env_name, env, task)
    return env_info


def load_datasets(paths, batch_size, env_indexes, n_task, seed=0, max_samples=None):
    # load datasets
    dataloaders = []
    for path in paths:
        data = pd.read_csv(path)
        feature_names = [column for column in data.columns if 's_' in column]
        target_names = [column for column in data.columns if 'a_' in column]
        X = torch.tensor(data[feature_names].to_numpy())
        y = torch.tensor(data[target_names].to_numpy())
        if max_samples is not None:
            X, y = X[:max_samples], y[:max_samples]
        dataset = torch.utils.data.TensorDataset(X, y)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)
    env_info = get_env_info(env_indexes, n_task, seed)
    return dataloaders, env_info


def train_agent(dataloaders, env_info, args):
    sample_env = list(env_info.values())[0][1]
    input_dim, action_space = sample_env.observation_space.shape[0], sample_env.action_space
    agent = BehaviorCloningAgent(input_dim, action_space, args.num_layers, args.hidden_dim)
    trainer = Trainer(args.run_name, agent, dataloaders, env_info, args)
    trainer.train()


def main():
    args = parse_args()
    wandb.init(project="igr4rl", entity="jasonyoo", config=args)
    wandb.define_metric("eval_step")
    wandb.define_metric("eval_return", step_metric="eval_step")
    args = DotDict(wandb.config)
    print(f"Training agent with settings: {args}")
    dataloaders, env_info = load_datasets(args.datasets, args.batch_size, args.env_indexes,
                                          args.n_task, seed=args.seed, max_samples=args.max_samples)
    train_agent(dataloaders, env_info, args)
    for info in env_info.values():
        env = info[1]
        env.close()


if __name__ == "__main__":
    main()
