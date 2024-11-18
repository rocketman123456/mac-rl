from torchrl.envs import GymEnv
from torchrl.envs import step_mdp

env = GymEnv("Pendulum-v1")

reset = env.reset()
print(reset)

reset_with_action = env.rand_action(reset)
print(reset_with_action)
print(reset_with_action["action"])

stepped_data = env.step(reset_with_action)
print(stepped_data)

data = step_mdp(stepped_data)
print(data)

rollout = env.rollout(max_steps=10)
print(rollout)

transition = rollout[3]
print(transition)
