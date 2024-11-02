from stable_baselines3 import PPO
from robot_env import RobotEnv

# train policy
env = RobotEnv()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)

# play policy
# obs = env.reset()
# for _ in range(1000):
#     action, _ = model.predict(obs)
#     obs, rewards, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()
