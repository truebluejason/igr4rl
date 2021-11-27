import numpy as np
import os
import torch
import wandb

from .nn import DeterministicPolicy


class BehaviorCloningAgent:
    def __init__(self, input_dim, action_space, num_layers, hidden_dim) -> None:
        self.ff = DeterministicPolicy(input_dim, action_space.shape[0], num_layers,
                                      hidden_dim, action_space)

    def __call__(self, state):
        return self.ff(state)

    def get_parameters(self):
        return {'ff': self.ff.parameters()}

    def compute_losses(self, X, y_target):
        y_pred = self.ff(X)
        ff_loss = ((y_pred - y_target) ** 2).sum()
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
        num_steps = 1
        env_index = 0
        best_avg_reward = -float('inf')
        losses = {}
        dataloaders = [iter(dataloader) for dataloader in self.dataloaders]
        while num_steps < self.num_steps:
            # switch dataloader every self.switch_env_every steps
            if num_steps % self.switch_env_every == 0:
                env_index = (env_index + 1) % len(dataloaders)
            dataloader = dataloaders[env_index]
            batch = next(dataloader, None)
            if batch is None:
                dataloaders[env_index] = iter(self.dataloaders[env_index])
                batch = next(dataloaders[env_index])

            # step optimizers
            X, y = batch[0].to(self.device).type(torch.float), batch[1].to(self.device)
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            batch_losses = self.agent.compute_losses(X, y)
            for name, loss in batch_losses.items():
                loss.backward()
                losses[name] = losses.get(name, []) + [loss.item()]
            for optimizer in self.optimizers.values():
                optimizer.step()

            # save and evaluate models once in a while
            if num_steps % 100 == 0:
                print(f"step {num_steps} loss: {losses[name][-1]}")
            eval_every = 1000
            if num_steps % eval_every == 0:
                log_dict = {}
                # log overall and per expert training loss
                task_avg_losses = []
                for name, task_losses in losses.items():
                    task_avg_losses.append(sum(task_losses)/len(task_losses))
                    log_dict[f'{name}_loss'] = task_avg_losses[-1]
                log_dict['train_loss'] = sum(task_avg_losses)/len(task_avg_losses)
                losses = {}

                # log overall and per environment performance
                task_avg_rewards = []
                for env_index, info in self.env_info.items():
                    avg_reward = self.evaluate(env_index, n_episodes=10)
                    log_dict[f'{info[0]}_return'] = avg_reward
                    task_avg_rewards.append(avg_reward)
                log_dict['eval_return'] = sum(task_avg_rewards)/len(task_avg_rewards)
                wandb.log(log_dict)
                print(f"REWARD: {log_dict['eval_return']}")

                # save checkpoints
                self.agent.save_checkpoint(self.run_name, suffix='latest', train_configs=self.args)
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    self.agent.save_checkpoint(
                        self.run_name, suffix='best', train_configs=self.args)

            num_steps += 1

    def evaluate(self, env_index, n_episodes=10):
        _, env, tasks = self.env_info[env_index]
        avg_reward = 0.
        for _ in range(n_episodes):
            env.set_task(tasks[np.random.randint(len(tasks))])
            state, episode_reward, episode_steps, done = env.reset(), 0, 0, False
            while not done:
                if self.args.render:
                    env.render()
                with torch.no_grad():
                    action = self.agent(torch.tensor(state).to(self.device).type(torch.float))
                state, reward, done, _ = env.step(action.cpu().numpy())
                episode_reward += reward
                episode_steps += 1
                done = done or episode_steps == env.max_path_length
            avg_reward += episode_reward
        avg_reward /= n_episodes
        return avg_reward
