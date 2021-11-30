import numpy as np
import os
import torch
import wandb

from .nn import DeterministicPolicy
from .utils import evaluate_on_env


class BehaviorCloningAgent:
    def __init__(self, input_dim, action_space, num_layers, hidden_dim) -> None:
        self.ff = DeterministicPolicy(input_dim, action_space.shape[0], num_layers,
                                      hidden_dim, action_space)

    def select_action(self, state, evaluate=False):
        # takes and returns numpy arrays but operates with tensors
        state = torch.tensor(state).type(torch.float)
        return self.ff(state.to(next(self.ff.parameters()).device)).cpu().numpy()

    def get_parameters(self):
        return {'ff': self.ff.parameters()}

    def compute_losses(self, X, y_target):
        y_pred = self.ff(X)
        ff_loss = ((y_pred - y_target) ** 2).sum()
        # Add VAE loss here. Maybe even add generative replay data here.
        return {'ff': ff_loss}

    def save_checkpoint(self, env_name, suffix="", ckpt_path=None,  train_configs=None):
        save_dir = f'agent/{env_name}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if ckpt_path is None:
            ckpt_path = f"{save_dir}/model_{suffix}_{train_configs.lr}"
        print('Saving models to {}'.format(ckpt_path))
        torch.save({'ff_state_dict': self.ff.state_dict(),
                    'train_configs': dict(train_configs)}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading models from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.ff.load_state_dict(checkpoint['ff_state_dict'])
            if evaluate:
                self.ff.eval()
            else:
                self.ff.train()

    def to(self, device):
        self.ff.to(device)


class Trainer:
    def __init__(self, run_name, agent, dataloaders, env_info, args):
        self.run_name = run_name
        self.agent = agent
        self.dataloaders = dataloaders
        self.env_info = env_info

        self.num_steps = args.num_steps
        self.switch_env_every = args.switch_env_every
        self.optimizers = {name: torch.optim.Adam(params, lr=args.lr)
                           for name, params in agent.get_parameters().items()}
        self.device = torch.device('cuda' if args.cuda else 'cpu')
        self.agent.to(self.device)
        self.args = args

    def train(self):
        num_steps = 0
        dataloader_index = 0
        best_avg_reward = -float('inf')
        losses_dict = {}  # Format: {<env name>: {<module name>: <list of losses>}}
        dataloaders = [iter(dataloader) for dataloader in self.dataloaders]

        while num_steps < self.num_steps:

            # switch dataloader every self.switch_env_every steps
            if (num_steps + 1) % self.switch_env_every == 0:
                dataloader_index = (dataloader_index + 1) % len(dataloaders)
            dataloader = dataloaders[dataloader_index]
            batch = next(dataloader, None)
            if batch is None:
                dataloaders[dataloader_index] = iter(self.dataloaders[dataloader_index])
                batch = next(dataloaders[dataloader_index])

            # step optimizers and update losses_dict
            X, y = batch[0].to(self.device).type(torch.float), batch[1].to(self.device)
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            batch_losses = self.agent.compute_losses(X, y)
            curr_env_name = list(self.env_info.values())[dataloader_index][0]
            for module_name, loss in batch_losses.items():
                loss.backward()
                env_losses_dict = losses_dict.get(curr_env_name, {})
                env_losses_dict[module_name] = [*env_losses_dict.get(module_name, []), loss.item()]
                losses_dict[curr_env_name] = env_losses_dict
            for optimizer in self.optimizers.values():
                optimizer.step()

            # save and evaluate models once in a while
            eval_every = 1000
            if num_steps % eval_every == 0:
                # log overall and per environment training loss and performance
                fmt_losses_dict = self.get_fmt_losses_dict(losses_dict)
                fmt_return_dict = self.evaluate()
                wandb.log({**fmt_losses_dict, **fmt_return_dict})
                losses_dict = {}
                print(f"Losses: {fmt_losses_dict}")
                print(f"Return: {fmt_return_dict}")

                # save checkpoints
                self.agent.save_checkpoint(self.run_name, suffix='latest', train_configs=self.args)
                if fmt_return_dict['eval_return'] > best_avg_reward:
                    best_avg_reward = fmt_return_dict['eval_return']
                    self.agent.save_checkpoint(
                        self.run_name, suffix='best', train_configs=self.args)

            num_steps += 1

    def get_fmt_losses_dict(self, losses_dict):
        # log per environment training loss for each module
        result_dict, module_avg_losses = {}, {}
        for (env_name, _, _) in self.env_info.values():
            env_loss_info = losses_dict.get(env_name, None)
            if env_loss_info is None:
                continue
            for module_name, module_losses in env_loss_info.items():
                module_loss = sum(module_losses)/len(module_losses)
                result_dict[f'{env_name}_{module_name}_loss'] = module_loss
                module_avg_losses[f'{module_name}_loss'] = \
                    module_avg_losses.get(f'{module_name}_loss', []) + [module_loss]
        return {**result_dict, **{k: sum(v)/len(v) for k, v in module_avg_losses.items()}}

    def evaluate(self):
        # log per environment performance
        return_dict, env_avg_rewards = {}, []
        for (env_name, env, tasks) in self.env_info.values():
            avg_reward, _ = evaluate_on_env(self.agent, env, tasks, n_episodes=50,
                                            random=False, collect_data=False, render=self.args.render)
            return_dict[f'{env_name}_return'] = avg_reward
            env_avg_rewards.append(avg_reward)
        return_dict['eval_return'] = sum(env_avg_rewards)/len(env_avg_rewards)
        return return_dict
