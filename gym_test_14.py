from __future__ import annotations

from framework.agent.reinforce import REINFORCE
from framework.env.vectored_mujoco_env import VectoredMujocoEnv
from framework.config.vectored_mujoco_cfg import VectoredMujocoCfg
from framework.runner.runner import PolicyRunner

if __name__ == "__main__":
    # plt.rcParams["figure.figsize"] = (10, 5)
    # Create and wrap the environment
    name = "Ant-v5"
    xml_file = "./mujoco_menagerie/unitree_go1/scene.xml"
    cfg = VectoredMujocoCfg(name, xml_file, 4, True)
    env = VectoredMujocoEnv(cfg)

    # Observation-space of InvertedPendulum-v4 (4)
    obs_space_dims = env.obs_space_dims
    # Action-space of InvertedPendulum-v4 (1)
    action_space_dims = env.action_space_dims

    agent = REINFORCE(cfg.device, obs_space_dims, action_space_dims)

    runner = PolicyRunner(agent, env)
    runner.learn()
