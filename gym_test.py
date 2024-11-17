"""
Author: Zhang Yufeng 759094438@qq.com
Date: 2024-11-17 21:47:03
LastEditors: Zhang Yufeng 759094438@qq.com
LastEditTime: 2024-11-17 21:47:05
FilePath: /mac-robot/mac-rl/gym_test.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
"""

import gymnasium as gym

# Initialise the environment
env = gym.make("LunarLander-v3", render_mode="human")

# Reset the environment to generate the first observation
observation, info = env.reset(seed=42)
for _ in range(1000):
    # this is where you would insert your policy
    action = env.action_space.sample()

    # step (transition) through the environment with the action
    # receiving the next observation, reward and if the episode has terminated or truncated
    observation, reward, terminated, truncated, info = env.step(action)

    # If the episode has ended then we can reset to start a new episode
    if terminated or truncated:
        observation, info = env.reset()

env.close()
