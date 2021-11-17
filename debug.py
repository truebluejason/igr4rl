import metaworld
import random


def run_env(env, tasks, t=250, render=False):
    # task = random.choice(mt1.train_tasks)

    for task_id in range(3):
        env.set_task(tasks[task_id])  # Set task

        obs = env.reset()  # Reset environment
        for t in range(500):
            if render:
                env.render()
            a = env.action_space.sample()  # Sample an action
            # Step the environoment with the sampled random action
            obs, reward, done, info = env.step(a)


mt10 = metaworld.MT10()  # Construct the benchmark, sampling tasks

training_envs = []
for name, env_cls in mt10.train_classes.items():
    env = env_cls()
    tasks = [task for task in mt10.train_tasks if task.env_name == name]
    training_envs.append((env, tasks))
    print(name)

for (env, tasks) in training_envs:
    run_env(env, tasks)
