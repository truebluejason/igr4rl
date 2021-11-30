import numpy as np
import torch


def evaluate_on_env(agent, env, tasks, n_episodes, random=False, collect_data=False, render=False):
    # Evaluate agent in a single environment on multiple tasks
    avg_reward = 0.
    state_action_info = {'state': [], 'action': []}
    for i in range(n_episodes):
        task_index = np.random.randint(len(tasks)) if random else i % len(tasks)
        env.set_task(tasks[task_index])
        state, episode_reward, episode_steps, done = env.reset(), 0, 0, False
        while not done:
            if render:
                env.render()
            with torch.no_grad():
                action = agent.select_action(state, evaluate=True)
            if collect_data:
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
