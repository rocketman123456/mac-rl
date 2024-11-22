from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import gymnasium as gym
from tqdm import tqdm
from algorithms.a2c import A2C

if __name__ == "__main__":
    # environment hyperparams
    n_envs = 3
    n_updates = 1000
    n_steps_per_update = 128
    randomize_domain = False

    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: gym.make(
                "Ant-v5",
                xml_file="./mujoco_menagerie/unitree_go1/scene.xml",
                forward_reward_weight=1,
                ctrl_cost_weight=0.05,
                contact_cost_weight=5e-4,
                healthy_reward=1,
                main_body=1,
                healthy_z_range=(0.195, 0.75),
                include_cfrc_ext_in_observation=True,
                exclude_current_positions_from_observation=False,
                reset_noise_scale=0.1,  # set to avoid policy overfitting
                frame_skip=25,  # set dt=0.05
                max_episode_steps=1000,  # kept at 1000
                # gravity=np.clip(np.random.normal(loc=-10.0, scale=1.0), a_min=-11.99, a_max=-0.01),
                # enable_wind=np.random.choice([True, False]),
                # wind_power=np.clip(np.random.normal(loc=15.0, scale=1.0), a_min=0.01, a_max=19.99),
                # turbulence_power=np.clip(np.random.normal(loc=1.5, scale=0.5), a_min=0.01, a_max=1.99),
                # max_episode_steps=600,
            )
            for i in range(n_envs)
        ]
    )

    # agent hyperparams
    gamma = 0.999
    lam = 0.95  # hyperparameter for GAE
    ent_coef = 0.01  # coefficient for the entropy bonus (to encourage exploration)
    actor_lr = 0.001
    critic_lr = 0.005

    obs_shape = envs.single_observation_space.shape[0]
    action_shape = envs.single_action_space.shape[0]

    # set the device
    use_cuda = False
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("mps" if torch.mps.is_available() else "cpu")

    # init the agent
    agent = A2C(obs_shape, action_shape, device, critic_lr, actor_lr, n_envs)

    # create a wrapper environment to save episode returns and episode lengths
    envs_wrapper = gym.wrappers.vector.RecordEpisodeStatistics(envs, buffer_length=n_envs * n_updates)

    critic_losses = []
    actor_losses = []
    entropies = []

    # use tqdm to get a progress bar for training
    for sample_phase in tqdm(range(n_updates)):
        # we don't have to reset the envs, they just continue playing
        # until the episode is over and then reset automatically

        # reset lists that collect experiences of an episode (sample phase)
        ep_value_preds = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_rewards = torch.zeros(n_steps_per_update, n_envs, device=device)
        ep_action_log_probs = torch.zeros(n_steps_per_update, n_envs, device=device)
        masks = torch.zeros(n_steps_per_update, n_envs, device=device)

        # at the start of training reset all envs to get an initial state
        if sample_phase == 0:
            states, info = envs_wrapper.reset(seed=42)

        # play n steps in our parallel environments to collect data
        for step in range(n_steps_per_update):
            # select an action A_{t} using S_{t} as input for the agent
            actions, action_log_probs, state_value_preds, entropy = agent.select_action(states)

            # perform the action A_{t} in the environment to get S_{t+1} and R_{t+1}
            states, rewards, terminated, truncated, infos = envs_wrapper.step(actions.cpu().numpy())

            ep_value_preds[step] = torch.squeeze(state_value_preds)
            ep_rewards[step] = torch.tensor(rewards, dtype=torch.float32, device=device)
            ep_action_log_probs[step] = torch.sum(action_log_probs, dim=1)

            # add a mask (for the return calculation later);
            # for each env the mask is 1 if the episode is ongoing and 0 if it is terminated (not by truncation!)
            masks[step] = torch.tensor([not term for term in terminated])

        # calculate the losses for actor and critic
        critic_loss, actor_loss = agent.get_losses(
            ep_rewards,
            ep_action_log_probs,
            ep_value_preds,
            entropy,
            masks,
            gamma,
            lam,
            ent_coef,
            device,
        )

        # update the actor and critic networks
        agent.update_parameters(critic_loss, actor_loss)

        # log the losses and entropy
        critic_losses.append(critic_loss.detach().cpu().numpy())
        actor_losses.append(actor_loss.detach().cpu().numpy())
        entropies.append(entropy.detach().mean().cpu().numpy())

    rolling_length = 20
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 5))
    fig.suptitle(
        f"Training plots for {agent.__class__.__name__} in the LunarLander-v3 environment \n \
                (n_envs={n_envs}, n_steps_per_update={n_steps_per_update}, randomize_domain={randomize_domain})"
    )

    # episode return
    axs[0][0].set_title("Episode Returns")
    episode_returns_moving_average = (
        np.convolve(
            np.array(envs_wrapper.return_queue).flatten(),
            np.ones(rolling_length),
            mode="valid",
        )
        / rolling_length
    )
    axs[0][0].plot(
        np.arange(len(episode_returns_moving_average)) / n_envs,
        episode_returns_moving_average,
    )
    axs[0][0].set_xlabel("Number of episodes")

    # entropy
    axs[1][0].set_title("Entropy")
    entropy_moving_average = np.convolve(np.array(entropies), np.ones(rolling_length), mode="valid") / rolling_length
    axs[1][0].plot(entropy_moving_average)
    axs[1][0].set_xlabel("Number of updates")

    # critic loss
    axs[0][1].set_title("Critic Loss")
    critic_losses_moving_average = np.convolve(np.array(critic_losses).flatten(), np.ones(rolling_length), mode="valid") / rolling_length
    axs[0][1].plot(critic_losses_moving_average)
    axs[0][1].set_xlabel("Number of updates")

    # actor loss
    axs[1][1].set_title("Actor Loss")
    actor_losses_moving_average = np.convolve(np.array(actor_losses).flatten(), np.ones(rolling_length), mode="valid") / rolling_length
    axs[1][1].plot(actor_losses_moving_average)
    axs[1][1].set_xlabel("Number of updates")

    plt.tight_layout()
    plt.show()
