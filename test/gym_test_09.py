import gymnasium as gym

if __name__ == "__main__":
    envs = gym.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="async")
    # envs = gym.vector.AsyncVectorEnv(
    #     [
    #         lambda: gym.make("Pendulum-v1", g=9.81),
    #         lambda: gym.make("Pendulum-v1", g=9.81),
    #     ]
    # )
    obs, infos = envs.reset(seed=42)
    _ = envs.action_space.seed(42)

    for _ in range(1000):
        # this is where you would insert your policy
        action = envs.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = envs.step(action)

        # If the episode has ended then we can reset to start a new episode
        if terminated.all() or truncated.all():
            observation, info = envs.reset()

    envs.close()
