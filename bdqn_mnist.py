# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqn_jaxpy
import os
import random
import time
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
    env_id: str = "mnist/0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = 10000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""

    feature_dim: int = 84
    """dimensions of feature representation layer"""

    # BLR parameters
    sigma: float = 0.5
    """prior variance for weights"""
    sigma_n: float = 0.8
    """noise variance"""
    posterior_update_freq: int = 1000
    """BLR posterior update frequency"""
    posterior_batch_size: int = 2000
    """batch size for posterior distribution update"""
    w_sample_freq: int = 100
    """Thompson sampling frequency"""
    alpha: float = 0.01
    """forgetting factor for BLR"""

def make_env(env_id, run_name):
    env = bsuite.load_and_record_to_csv(env_id, results_dir=f'runs/{run_name}')
    gym_env = gym_wrapper.GymFromDMEnv(env)
    return gym_env


# ALGO LOGIC: initialize agent here:
class FeatureNetwork(nn.Module):
    # action_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(120)(x)
        x = nn.relu(x)
        x = nn.Dense(args.feature_dim)(x)
        x = nn.relu(x)
        # x = nn.Dense(self.action_dim)(x)
        return x


class BayesianLinearRegression:
    def __init__(self, feature_dim, action_dim, sigma, sigma_n, alpha):
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.sigma = sigma
        self.sigma_n = sigma_n
        self.alpha = alpha

    def update_posterior(self, phi, actions, targets, PhiPhiT, PhiY):
        # phi: (batch_dim, feature_dim)
        # actions: (batch_dim,)
        # targets: (batch_dim,)

        # forgetting factor
        PhiPhiT = (1-self.alpha) * PhiPhiT
        PhiY = (1-self.alpha) * PhiY

        one_hot_actions = jax.nn.one_hot(actions, num_classes=self.action_dim) # (batch_dim, action_dim)

        phi_a = phi[:, None, :] * one_hot_actions[:, :, None] # (batch_dim, action_dim, feature_dim)
        # phi_a is either equal to phi_a if the action was taken or 0 if not.
        targets_a = targets[:, None] * one_hot_actions # (batch_dim, action_dim)

        PhiPhiT = PhiPhiT + jnp.einsum('bai,baj->aij', phi_a, phi_a) # (action_dim, feature_dim, feature_dim)
        PhiY = PhiY + jnp.einsum('bai, ba-> ai', phi_a, targets_a) # (action_dim, feature_dim)

        # Cov_W
        Xi_a = jax.vmap(jnp.linalg.inv, in_axes=0, out_axes=0)(PhiPhiT / jnp.power(self.sigma_n, 2) + jnp.eye(self.feature_dim)[None, :] / jnp.power(self.sigma, 2)) # (action_dim, feature_dim, feature_dim)

        E_W = jnp.einsum('aij,aj->ai', Xi_a, PhiY) / jnp.power(self.sigma_n, 2) # (action_dim, feature_dim)

        return E_W, Xi_a, PhiPhiT, PhiY

    def sample_weights(self, rng_key, E_W, Cov_W):
        keys = jax.random.split(rng_key, self.action_dim)
        def sample_weight_single(rng_key, E_W_a, Cov_W_a):
            L = jnp.linalg.cholesky((Cov_W_a + Cov_W_a.T) / 2)
            z = jax.random.normal(rng_key, (self.feature_dim,))
            return E_W_a + L @ z

        return jax.vmap(sample_weight_single)(keys, E_W, Cov_W)

class TrainState(TrainState):
    target_params: flax.core.FrozenDict
    blr_params: flax.core.FrozenDict
    target_blr_params: flax.core.FrozenDict
    key: jax.Array

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
        for key in ['env_id', 'sigma', 'sigma_n', 'posterior_update_freq', 'posterior_batch_size', 'w_sample_freq', 'alpha', 'seed']:
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

    obs = env.reset()
    obs_flattened = obs.flatten()
    feature_network = FeatureNetwork()

    keys = jax.random.split(key, 3)
    key = keys[0]

    action_dim = env.action_spec().num_values
    feature_dim = args.feature_dim

    E_W = 0.01 * jax.random.normal(keys[1], shape=(action_dim, feature_dim))
    sampled_W = 0.01 * jax.random.normal(keys[2], shape=(action_dim, feature_dim))
    Cov_W = jnp.zeros((action_dim, feature_dim, feature_dim)) + jnp.eye(feature_dim)[None, :, :]

    PhiPhiT = jnp.zeros((action_dim, feature_dim, feature_dim))
    PhiY = jnp.zeros((action_dim, feature_dim))

    blr = BayesianLinearRegression(
        feature_dim=args.feature_dim,
        action_dim=action_dim,
        sigma=args.sigma,
        sigma_n=args.sigma_n,
        alpha=args.alpha
    )

    q_state = TrainState.create(
        apply_fn=feature_network.apply,
        params=feature_network.init(q_key, obs_flattened),
        blr_params=flax.core.freeze({
            'E_W': E_W,
            'Cov_W': Cov_W,
            'PhiPhiT': PhiPhiT,
            'PhiY': PhiY,
            'sampled_W': sampled_W
        }),
        target_params=feature_network.init(q_key, obs_flattened),
        target_blr_params=flax.core.freeze({
            'E_W': E_W,
            'Cov_W': Cov_W,
            'PhiPhiT': PhiPhiT,
            'PhiY': PhiY,
            'sampled_W': sampled_W
        }),
        tx=optax.adam(learning_rate=args.learning_rate),
        key=key
    )

    feature_network.apply = jax.jit(feature_network.apply)
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

        targets = compute_targets(q_state, next_observations, rewards, dones)
        def mse_loss(params):
            q_pred = feature_network.apply(params, observations)
            q_pred = q_pred @ q_state.blr_params['E_W'].T # (batch_size, num_actions)
            q_pred = q_pred[jnp.arange(q_pred.shape[0]), actions.squeeze()]  # (batch_size,)
            return ((q_pred - targets) ** 2).mean(), q_pred

        (loss_value, q_pred), grads = jax.value_and_grad(mse_loss, has_aux=True)(q_state.params)
        q_state = q_state.apply_gradients(grads=grads)

        return loss_value, q_pred, q_state


    @jax.jit
    def compute_targets(q_state, next_observations, rewards, dones):
        phi_next_online = feature_network.apply(q_state.params, next_observations)
        q_next_online = phi_next_online @ q_state.blr_params['sampled_W'].T
        actions_online = jnp.argmax(q_next_online, axis=1)
        phi_next_target = feature_network.apply(q_state.target_params, next_observations)

        q_next_target = phi_next_target @ q_state.target_blr_params['E_W'].T
        q_next_target = jnp.take_along_axis(q_next_target, actions_online[:, None], axis=1).squeeze()
        targets = rewards + (1 - dones) * args.gamma * q_next_target

        return targets

    start_time = time.time()

    episodic_reward = 0
    rewards_sum = 0
    episodes = 0
    exploration_moves = 0

    # TRY NOT TO MODIFY: start the game
    obs = env.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        obs_flattened = obs.flatten()
        q_diff = 0
        phi = feature_network.apply(q_state.params, obs_flattened)
        q_values = phi @ q_state.blr_params['sampled_W'].T  # Thompson sampling
        actions = q_values.argmax(axis=-1)
        actions = jax.device_get(actions)

        expected_q_values = phi @ q_state.blr_params['E_W'].T
        expected_actions = expected_q_values.argmax(axis=-1)

        if actions != expected_actions:
            q_diff = q_values[actions] - expected_q_values[expected_actions]
            exploration_moves += 1

        expected_q_diff = expected_q_values[expected_actions] - expected_q_values[actions]

        uncertainty = phi @ q_state.blr_params['Cov_W'][actions].squeeze() @ phi.T

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, infos = env.step(actions)

        episodic_reward += rewards

        if terminations:
            if global_step % 100 == 0:
                writer.add_scalar("rewards/episodic_reward", episodic_reward, global_step)
                writer.add_scalar("exploration/exploration_moves", exploration_moves, global_step)
                writer.add_scalar("exploration/q_diff", np.array(q_diff), global_step)
                writer.add_scalar("exploration/expected_q_diff", np.array(expected_q_diff), global_step)
                writer.add_scalar("exploration/uncertainty", uncertainty.squeeze().item(), global_step)
                writer.add_scalar("exploration/step_q_values", q_values.max().item(), global_step)
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
                    data.actions.numpy().squeeze(),
                    data.next_observations.numpy(),
                    data.rewards.flatten().numpy(),
                    data.dones.flatten().numpy(),
                )

                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", jax.device_get(loss), global_step)
                    writer.add_scalar("losses/q_values", jax.device_get(old_val).mean(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

            if global_step % args.posterior_update_freq == 0:
                data = rb.sample(args.posterior_batch_size)
                targets = compute_targets(q_state,
                                          data.next_observations.numpy().reshape(args.posterior_batch_size, -1),
                                          data.rewards.flatten().numpy(),
                                          data.dones.flatten().numpy())
                phi = feature_network.apply(q_state.params, data.observations.numpy().reshape(args.posterior_batch_size, -1))
                E_W, Cov_W, PhiPhiT, PhiY = blr.update_posterior(phi, data.actions.numpy().squeeze(), targets,
                                                                 q_state.blr_params['PhiPhiT'],
                                                                 q_state.blr_params['PhiY'])

                q_state = q_state.replace(
                    blr_params=flax.core.freeze({
                        'E_W': E_W,
                        'Cov_W': Cov_W,
                        'PhiPhiT': PhiPhiT,
                        'PhiY': PhiY,
                        'sampled_W': q_state.blr_params['sampled_W']
                    })
                )

            # update target network
            if global_step % args.target_network_frequency == 0:
                q_state = q_state.replace(
                    # target_params=optax.incremental_update(q_state.params, q_state.target_params, args.tau),
                    target_params=q_state.params,
                    target_blr_params = q_state.blr_params
                )

        if global_step % args.w_sample_freq == 0:
            key, w_key = jax.random.split(q_state.key, 2)
            sampled_W = blr.sample_weights(w_key, q_state.blr_params['E_W'], q_state.blr_params['Cov_W'])
            q_state = q_state.replace(
                blr_params=flax.core.freeze({
                    'E_W': q_state.blr_params['E_W'],
                    'Cov_W': q_state.blr_params['Cov_W'],
                    'PhiPhiT': q_state.blr_params['PhiPhiT'],
                    'PhiY': q_state.blr_params['PhiY'],
                    'sampled_W': sampled_W
                }),
                key=key
            )

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        with open(model_path, "wb") as f:
            f.write(flax.serialization.to_bytes(q_state.params))
        print(f"model saved to {model_path}")

    if args.track:
        wandb.log({"score" : rewards_sum/episodes})

    env.close()
    writer.close()