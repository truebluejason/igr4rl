import argparse
import datetime
import itertools
import metaworld
import numpy as np
import os
import torch
import wandb

from soft_actor_critic import SAC, ReplayMemory
from utils import DotDict

# os.environ["WANDB_MODE"] = "offline"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-index', default=0, type=int,
                        help='Index of the MT10 environment to train on')
    parser.add_argument('--n-task', default=50, type=int,
                        help='Number of tasks to run on for the environment')
    parser.add_argument('--render', action="store_true",
                        help='Render evaluation episodes (default: False)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                              term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=0, metavar='N',
                        help='random seed (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                        help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    return parser.parse_args()


def get_mt10_env(env_index, n_task):
    mt10 = metaworld.MT10()
    env_name, env = list(mt10.train_classes.items())[env_index]
    tasks = [task for task in mt10.train_tasks if task.env_name == env_name]
    return env_name, env(), tasks[:n_task]


def set_seed(seed, env):
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def evaluate_agent(env, tasks, agent, n_episodes, render):
    avg_reward = 0.
    state_action_info = {'state': [], 'action': []}
    for _ in range(n_episodes):
        env.set_task(tasks[np.random.randint(len(tasks))])
        state, episode_reward, episode_steps, done = env.reset(), 0, 0, False
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            state_action_info['state'].append(state.tolist())
            state_action_info['action'].append(action.tolist())
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
            episode_steps += 1
            done = done or episode_steps == env.max_path_length
        avg_reward += episode_reward
    avg_reward /= n_episodes
    return avg_reward, state_action_info


def train_agent(env, tasks, env_name, args):
    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Memory
    memory = ReplayMemory(args.replay_size, args.seed)

    # Training Loop
    total_numsteps = 0
    updates = 0
    best_avg_reward = -float('inf')

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        env.set_task(tasks[np.random.randint(len(tasks))])
        state = env.reset()

        while not done:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                        memory, args.batch_size, updates)
                    wandb.log({'critic_1': critic_1_loss, 'critic_2': critic_2_loss, 'policy': policy_loss,
                              'entropy_loss': ent_loss, 'entropy_temprature/alpha': alpha})
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            # mask = 1 if episode_steps == env.max_path_length else float(not done)
            mask = done

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state
            done = done or episode_steps == env.max_path_length

        if total_numsteps > args.num_steps:
            break

        wandb.log({'train_return': episode_reward, 'train_step': i_episode})
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(
            i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))

        eval_every = 25
        if i_episode % eval_every == 0:
            n_episode = 10
            avg_reward, _ = evaluate_agent(
                env, tasks, agent, n_episodes=n_episode, render=args.render)
            wandb.log({'eval_return': avg_reward, 'eval_step': i_episode // eval_every})
            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(n_episode, round(avg_reward, 2)))
            print("----------------------------------------")

            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save_checkpoint(env_name, suffix='best', train_configs=args)
            agent.save_checkpoint(env_name, suffix='latest', train_configs=args)


if __name__ == "__main__":
    args = parse_arguments()
    wandb.init(project="igr4rl", entity="jasonyoo", config=args)
    wandb.define_metric("train_step")
    wandb.define_metric("train_return", step_metric="train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("eval_return", step_metric="eval_step")
    args = DotDict(wandb.config)
    env_name, env, tasks = get_mt10_env(args.env_index, args.n_task)
    print(f"Training SAC on {env_name} with settings: {args}")
    set_seed(args.seed, env)
    train_agent(env, tasks, env_name, args)
    env.close()
