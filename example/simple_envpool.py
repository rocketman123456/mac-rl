import envpool
import numpy as np

# Set the number of parallel environments
num_envs = 16

# Create the EnvPool environment for MuJoCo (e.g., "Hopper-v2")
env = envpool.make("Hopper-v2", num_envs=num_envs)

# Reset the environments
obs = env.reset()

# Run a simple simulation loop
for step in range(100):
    # Random actions for demonstration
    actions = np.random.uniform(-1, 1, size=(num_envs, env.action_space.shape[0]))

    # Step the environment
    obs, rewards, dones, infos = env.step(actions)

    # Print the observations and rewards for the first environment
    print(f"Step {step}: Observations {obs[0]}, Rewards {rewards[0]}, Done {dones[0]}")

# Close the environment
env.close()
