import dm_control.suite
import numpy as np

# Load the environment
env = dm_control.suite.load(domain_name="cartpole", task_name="swingup")

# Training loop
num_episodes = 10
for episode in range(num_episodes):
    time_step = env.reset()
    total_reward = 0

    while not time_step.last():
        action = np.random.uniform(low=-1.0, high=1.0, size=env.action_spec().shape)  # Random action
        time_step = env.step(action)

        total_reward += time_step.reward

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")
