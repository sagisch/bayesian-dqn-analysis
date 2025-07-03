# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass

import flax
import flax.linen as nn
import gymnasium as gym
import bsuite
from bsuite.utils import gym_wrapper
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
import matplotlib.pyplot as plt

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "deep_sea/0"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training"""


def make_env(env_id, run_name):
    env = bsuite.load_and_record_to_csv(env_id, results_dir=f'runs/{run_name}')
    gym_env = gym_wrapper.GymFromDMEnv(env)
    return gym_env


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(84)(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim)(x)
        return x


class TrainState(TrainState):
    target_params: flax.core.FrozenDict


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

        config = wandb.config
        for key in ['env_id', 'exploration_fraction', 'start_e', 'end_e', 'seed']:
            setattr(args, key, getattr(config, key))
            print("config value: ", getattr(config, key), " arg value: ", getattr(args, key))

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    key, q_key = jax.random.split(key, 2)

    # env setup
    env = make_env(args.env_id, run_name)

    # assert isinstance(env.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    obs = env.reset()
    obs_flattened = obs.flatten()
    action_dim = env.action_spec().num_values

    q_network = QNetwork(action_dim=action_dim)
    q_state = TrainState.create(
        apply_fn=q_network.apply,
        params=q_network.init(q_key, obs_flattened),
        target_params=q_network.init(q_key, obs_flattened),
        tx=optax.adam(learning_rate=args.learning_rate),
    )

    q_network.apply = jax.jit(q_network.apply)
    # This step is not necessary as init called on same observation and key will always lead to same initializations
    q_state = q_state.replace(target_params=optax.incremental_update(q_state.params, q_state.target_params, 1))

    observation_space = gym.spaces.Box(
        env.observation_space.low,
        env.observation_space.high,
        env.observation_space.shape,
        env.observation_space.dtype
    )

    action_space = gym.spaces.Discrete(
        env.action_space.n,
        args.seed+1,
        env.action_space.start
    )

    rb = ReplayBuffer(
        args.buffer_size,
        observation_space,
        action_space,
        "cpu",
        handle_timeout_termination=False,
    )

    @jax.jit
    def update(q_state, observations, actions, next_observations, rewards, dones):
        observations = observations.reshape(args.batch_size,-1)
        next_observations = next_observations.reshape(args.batch_size,-1)

        q_next_target = q_network.apply(q_state.target_params, next_observations)  # (batch_size, num_actions)
        q_next_target = jnp.max(q_next_target, axis=-1)  # (batch_size,)
        next_q_value = rewards + (1 - dones) * args.gamma * q_next_target

        def mse_loss(params):
            q_pred = q_network.apply(params, observations)  # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - next_q_value) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)
        return loss_value, q_pred, q_state

    start_time = time.time()

    episodic_reward = 0
    rewards_sum = 0
    episodes = 0
    exploration_moves = 0

    # TRY NOT TO MODIFY: start the game
    obs = env.reset()

    obs_dim = obs.shape[0]
    state_coverage = jnp.zeros(shape=(obs_dim, obs_dim))

    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        state_coverage += obs
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        obs_flattened = obs.flatten()
        q_values = q_network.apply(q_state.params, obs_flattened)
        max_actions = q_values.argmax(axis=-1)

        q_diff = 0
        if random.random() < epsilon:
            actions = np.array(random.randrange(2))
            exploration_moves += 1
            q_diff = q_values[max_actions] - q_values[actions]
        else:
            actions = jax.device_get(max_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, infos = env.step(actions)

        episodic_reward += rewards

        if global_step % 100 == 0:
            writer.add_scalar("exploration/q_diff", np.array(q_diff).item(), global_step)

        if terminations:
            writer.add_scalar("rewards/episodic_reward", episodic_reward, global_step)
            writer.add_scalar("exploration/exploration_moves", exploration_moves, global_step)
            rewards_sum += episodic_reward
            episodes += 1
            episodic_reward = 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                # perform a gradient-descent step
                loss, old_val, q_state = update(
                    q_state,
                    data.observations.numpy(),
                    data.actions.numpy(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau)
                )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

    print(rewards_sum/episodes)
    if args.track:
        wandb.log({"score" : rewards_sum/episodes})
        wandb.run.summary["state-matrix"] = state_coverage.tolist()
        plt.figure(figsize=(12, 9), dpi=600)
        sns.heatmap(state_coverage, annot=False, cmap="rocket_r")
        plt.title("State Coverage")
        wandb.log({"state-coverage": wandb.Image(plt)})

    env.close()
    writer.close()
