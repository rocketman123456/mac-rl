from __future__ import annotations

from framework.env.vectored_gym_env import VectoredGymEnv
from framework.config.vectored_gym_cfg import VectoredGymCfg
from framework.agent.reinforce import REINFORCE
from framework.runner.runner import PolicyRunner

if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = (10, 5)
    # Create and wrap the environment
    name = "InvertedPendulum-v5"
    cfg = VectoredGymCfg(name, 4, True)
    env = VectoredGymEnv(cfg)

    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.obs_space_dims
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space_dims

    agent = REINFORCE(cfg.device, obs_space_dims, action_space_dims)

    runner = PolicyRunner(agent, env)
    runner.learn()
