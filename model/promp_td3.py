import torch.distributions
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from .td3_policy import TD3Policy
import matplotlib.pyplot as plt
from model.promp_policy import DetPMPWrapper

import io
import pathlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th

from stable_baselines3.common import logger
from .base_class import BaseAlgorithm
from .base_policy import BasePolicy
from .td3_policy import TD3Policy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from .Replay_buffer_with_step import ReplayBufferStep
import gym

class ProMPTD3(BaseAlgorithm):

    def __init__(
        self,
        policy: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        learning_starts: int = 2000,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "episode"),
        gradient_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        data_path: str = None,
    ):

        super(ProMPTD3, self).__init__(
            policy=policy,
            env=env,
            policy_base=TD3Policy,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=None,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage

        # Remove terminations (dones) that are due to time limit
        self.remove_time_limit_termination = False

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments

        # For gSDE only
        self.use_sde_at_warmup = False
        self.episode_timesteps = 0

        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        self.basis_num = 10
        self.dof = 5
        self.noise_sigma = 0.3
        self.actor_lr = 0.00001

        self.mean = 0.1 * th.ones(self.basis_num*self.dof)#torch.randn(25,)
        self.promp_params = ((self.mean).reshape(self.basis_num, self.dof)).to(device="cuda")

        self.data_path = data_path
        self.best_model = -9000000

        self.max_episode_steps = 200
        self.ls_number = 0
        self.action_noise = action_noise
        self.eval_freq = 1000

        self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.replay_buffer = ReplayBufferStep(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()
        self._create_aliases()

    def _create_aliases(self) -> None:
        actor_kwargs = {"policy_kwargs": {"p_gains": 1, "d_gains": 0.1}}

        self.actor = DetPMPWrapper(self.env, num_dof=self.dof, num_basis=self.basis_num, duration=4, width=0.025,
                                          policy_type="motor", weights_scale=1, zero_start=False, step_length=self.max_episode_steps,
                                          policy_kwargs=actor_kwargs, noise_sigma=self.noise_sigma)

        self.actor_target = DetPMPWrapper(self.env, num_dof=self.dof, num_basis=self.basis_num, duration=4, width=0.025,
                                          policy_type="motor", weights_scale=1, zero_start=False, step_length=self.max_episode_steps,
                                          policy_kwargs=actor_kwargs, oise_sigma=self.noise_sigma)

        self.actor.mp.weights = self.promp_params
        (self.actor.mp.weights).requires_grad = True
        self.actor_target.mp.weights = self.actor.mp.weights #(self.promp_params * self.tau) # + (1 - self.tau) * self.actor_target.mp.weights)
        self.actor.update()
        self.actor_target.update()

        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

        self.actor_optimizer = th.optim.Adam([self.actor.mp.weights], lr=self.actor_lr)

        self.weights_noise = False


    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        # Update learning rate according to lr schedule
        self.reward_with_noise = self.env.envs[0].rewards_intotal

        self.eval_reward = self.actor.eval_rollout(self.env, self.actor.mp.weights.reshape(-1,5))
        if self.best_model < self.env.envs[0].rewards_intotal:
            self.best_model = self.env.envs[0].rewards_intotal
            np.save(self.data_path + "/best_model.npy", self.actor.mp.weights.cpu().detach().numpy())
        np.save(self.data_path + "/algo_mean.npy", self.actor.mp.weights.cpu().detach().numpy())
        self._update_learning_rate([self.critic.optimizer])

        #if self.num_timesteps == 2200 or self.num_timesteps == 2001:
        #    self.polt_trajectory()
        #elif self.num_timesteps % 2.e4 == 0:
        #    self.polt_trajectory()

        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():

                # Select action according to policy and add clipped noise
                noise = replay_data.actions.clone().data.normal_(0, self.target_policy_noise)
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                next_step = replay_data.next_steps

                next_actions = (self.actor_target.predict_action(next_step, replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions, (next_step+1)/self.max_episode_steps),
                                       dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            current_q_values = self.critic(replay_data.observations, replay_data.actions, ((replay_data.steps+1)/self.max_episode_steps))

            # Compute critic loss
            critic_loss = sum([F.mse_loss(current_q,  target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critics
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()

            # Delayed policy updates
            if self._n_updates % self.policy_delay == 0:
                # Compute actor loss
                act = self.actor.predict_action(replay_data.steps, replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, act,  (replay_data.steps+1)/self.max_episode_steps).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                self.actor_target.mp.weights = (self.actor.mp.weights * self.tau + (1 - self.tau) * self.actor_target.mp.weights).to(device="cuda")
                self.actor.update()
                self.actor_target.update()
        if self.num_timesteps % 800 == 0:
            print("weights", self.actor.mp.weights)
        #print("self.cov@simple", self.sample)
        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if len(actor_losses) > 0:
            logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("eval/noise_reward", self.reward_with_noise) #self.env.venv.envs[0].rewards_intotal)
        logger.record("train/actor_learning_rate", self.actor_lr)
        logger.record("train/gradient_steps", gradient_steps)
        logger.record("train/noise_sigma", self.noise_sigma)
        logger.record("train/num_basis", self.basis_num)
        logger.record("eval/mean_reward", self.eval_reward)

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "run",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
    ) -> "OffPolicy":

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps,
            tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps > 0 else rollout.episode_timesteps
                self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)

        callback.on_training_end()

        return self


    def _sample_action(
        self, episode_timesteps: int, action_noise: Optional[ActionNoise] = None) -> Tuple[np.ndarray, np.ndarray]:

        unscaled_action = self.actor.get_action(episode_timesteps)
        #unscaled_action = self.actor.get_action(episode_timesteps, weights_noise=self.weights_noise)
        #print("action", action_noise)
        #unscaled_action = self.actor.get_action(episode_timesteps, action_noise=self.action_noise)
        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action).reshape(1, 5)  # 2.0 * ((action - low) / (high - low)) - 1.0
            #if action_noise is not None:
            #    scaled_action = np.clip(scaled_action.reshape(1, 5) + action_noise(), -1, 1)
            #action_noise = NormalActionNoise(mean=np.zeros(scaled_action.shape), sigma=0 * np.ones(scaled_action.shape))
            #scaled_action = np.clip(scaled_action+action_noise().reshape(1,5), -1, 1)
            scaled_action = np.clip(scaled_action, -1, 1)
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action) #low + (0.5 * (scaled_action + 1.0) * (high - low))
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = np.clip(unscaled_action+action_noise(), -1, 1)
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        logger.record("time/fps", fps)
        logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)

    def _store_transition(
            self,
            replay_buffer: ReplayBufferStep,
            buffer_action: np.ndarray,
            new_obs: np.ndarray,
            reward: np.ndarray,
            done: np.ndarray,
            infos: List[Dict[str, Any]],
            steps: np.ndarray,
            next_steps: np.ndarray
    ) -> None:

        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, steps, next_steps)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts(
            self,
            env: VecEnv,
            callback: BaseCallback,
            train_freq: TrainFreq,
            replay_buffer: ReplayBufferStep,
            action_noise: Optional[ActionNoise] = None,
            learning_starts: int = 0,
            log_interval: Optional[int] = None,
    ) -> RolloutReturn:

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True
        if self.episode_timesteps == 0:
            self.plot_vel_with_noise = np.zeros((200, 5))
            self.plot_pos_with_noise = np.zeros((200, 5))
            self.actor.update()
            if self.weights_noise:
                self.actor.update_tra_with_noise((action_noise))

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes) or \
                self.ls_number < self.learning_starts:  # TrainFreq(frequency=1, unit=<TrainFrequencyUnit.STEP: 'step'>) 0 0
            # loop for episode
            done = False
            episode_reward = 0.0

            while not done:  # loop for steps during one episode (timesteps plus one)

                # Select action randomly or according to policy
                ### TODO: only use one WEIGHT in the beginning
                action, buffer_action = self._sample_action(self.episode_timesteps, action_noise)

                # Rescale and perform actionp
                action = action.reshape(action.shape[0], -1)
                new_obs, reward, done, infos = env.step(action)
                self.plot_pos_with_noise[self.episode_timesteps] = new_obs[:, -10:-5].reshape(-1)
                self.plot_vel_with_noise[self.episode_timesteps] = new_obs[:, -5:].reshape(-1)

                self.num_timesteps += 1
                self.episode_timesteps += 1
                self.ls_number += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                next_step = self.episode_timesteps
                if self.episode_timesteps == 200:
                    next_step = 199
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos,
                                       self.episode_timesteps - 1, next_step)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break

            if done:
                num_collected_episodes += 1
                self._episode_num += 1
                # print("")
                episode_rewards.append(episode_reward)
                total_timesteps.append(self.episode_timesteps)
                self.episode_timesteps = 0

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                if self.weights_noise:
                    self.actor.update_tra_with_noise((action_noise))

                # self.actor.update()

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)


    def _convert_train_freq(self) -> None:

        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")
                #train_freq = (2000, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46
        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and self.replay_buffer is not None
            and (self.replay_buffer.full or self.replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (self.replay_buffer.pos - 1) % self.replay_buffer.buffer_size
            self.replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, log_path, reset_num_timesteps, tb_log_name
        )

    def polt_trajectory(self):
        print("save pos and vel pics")
        for i in range(5):
            plt.plot(self.actor.plot_pos[:, i], label='without noise + reward:' + str(self.eval_reward))
            plt.plot(self.plot_pos_with_noise[:, i],
                     label='with noise + reward:' + str(self.reward_with_noise)) # label_name(name))
            plt.legend()
            plt.title('timestep: ' + str(self.num_timesteps) + f', position_joint_{i}')
            plt.savefig(f'plots_actionnoise_3/timestep: ' + str(self.num_timesteps) + f'_position_joint_{i}')
            plt.cla()

            plt.plot(self.actor.plot_vel[:, i], label='without noise + reward:' + str(self.eval_reward))
            plt.plot(self.plot_vel_with_noise[:, i], label='with noise + reward:' + str(self.reward_with_noise))  # label_name(name))
            plt.legend()
            plt.title('timestep: ' + str(self.num_timesteps) + f', velocity_joint_{i}')
            plt.savefig('plots_actionnoise_3/timestep: ' + str(self.num_timesteps) + f'_velocity_joint_{i}')
            plt.cla()




