from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
import matplotlib.pyplot as plt

import io
import pathlib
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import numpy as np
import torch as th

from stable_baselines3.common import logger
from .base_algorithm import BaseAlgorithm
from .td3_policy import TD3Policy
from .detpmp_wrapper import DetPMPWrapper

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from .replay_buffer import ReplayBufferStep
import gym
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from .contextual import ContextNN



class EpisodicTD3(BaseAlgorithm):

    """
    This class is the interface for the users using Episodic TD3 algorithm.

    :param critic: the critic network in Episodic TD3  (MlpPolicy, CnnPolicy, ...).
    :param env: for any regular OpenAI Gym environment.
    :param initial_promp_params: the initial value of ProMP parameters, it can be int, float or tensor
    :param basis_num: the number of Gaussian Basis Functions.
    :param learning_start_episodes: how many episodes to collect before learning starts.
    :param critic_learning_rate: the learning rate of the critic network.
    :param actor_lraning_rate: the learning rate of the ProMP actor.
    :param buffer_size: size of the replay buffer.
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per iteration. The Q values will be updated policy_delay more often (update every training step).
    :param noise_sigma: the standard deviation value of exploration noise
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param critic_network_kwargs: additional arguments to be passed to the critic network on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param data_path: the path to save model and tensorboard data.
    """

    def __init__(
        self,
        critic: Union[str, Type[TD3Policy]],
        env: Union[GymEnv, str],
        schedule=None,
        initial_promp_params: th.Tensor = None,
        basis_num: int = 10,
        learning_start_episodes: int = 0,
        critic_learning_rate: Union[float, Schedule] = 1e-3,
        actor_learning_rate:  Union[float, Schedule] = 1e-3,
        buffer_size: int = int(1e6),
        tau: float = 0.005,
        gamma: float = 0.99,
        policy_delay: int = 2,
        noise_sigma: float = 0.1,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        critic_network_kwargs: Dict[str, Any] = None,
        promp_policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        data_path: str = None,
        contextual: bool = False,
        context_hidden_layer: int = 256,
        weight_noise_judge: bool = False,
        weight_noise: int = 1,
        pos_traj_steps: int =32,
    ):


        # Initialize TD3 critic network, for details please check OpenAI Stable baselines3
        super(EpisodicTD3, self).__init__(
            policy=critic,
            env=env,
            policy_base=TD3Policy,
            learning_rate=critic_learning_rate,
            policy_kwargs=critic_network_kwargs,
            tensorboard_log=data_path,
            verbose=verbose,
            device=device,
            seed=seed,
            supported_action_spaces=None,
        )

        self.eval_numsteps = 0


        # Setup the default setting of stable baselines3, for details please check OpenAI Stable baselines3
        self.remove_time_limit_termination = False
        self.actor = None
        self.replay_buffer = None
        self.use_sde_at_warmup = False
        self.episode_timesteps = 0
        self.optimize_memory_usage = False

        # learning rate and noise schedule
        if schedule is not None:
            self.need_schedule = True
            self.schedule = schedule
        else:
            self.need_schedule = False

        # Path for saving the model
        self.data_path = data_path

        self.buffer_size = buffer_size
        self.tau = tau
        self.policy_delay = policy_delay
        self.gamma = gamma

        # Noise for target policy smoothing regularization
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        # Environment episode length
        self.pos_traj_steps = pos_traj_steps

        if "Meta" in str(env):
            self.env.random_init = False
            self.max_episode_steps = self.env.max_path_length
        else:
            self.max_episode_steps = self.env.max_episode_steps

        # Save train freq parameter, will be converted later to TrainFreq object
        # Set the batch size, training frequency and gradient steps equal to the length of one episode
        self.train_freq = self.max_episode_steps  # How many gradient steps to do after each rollout
        self.gradient_steps = self.max_episode_steps  # Update the model every ``train_freq`` timesteps.
        self.batch_size = self.max_episode_steps  # How many data to use in training

        # How many timesteps of the model to collect transitions for before learning starts.
        self.learning_starts = self.max_episode_steps * learning_start_episodes
        self.ls_number = 0 # Counting the sample timesteps from the start

        # Setup the learning rate of the optimizer which updates the ProMP weights of Gaussian Basis Function
        self.actor_learning_rate = actor_learning_rate

        # Set the parameters of ProMP wrapper
        self.basis_num = basis_num
        self.dof = env.action_space.shape[0]

        # set the exploration noise
        self.noise_sigma = noise_sigma

        self.weight_noise_judge = weight_noise_judge
        if self.weight_noise_judge == True:
            self.weight_noise = weight_noise
            self.param_noise = NormalActionNoise(mean=np.zeros(self.dof * self.basis_num),
                                                 sigma=self.weight_noise * np.ones(self.dof * self.basis_num))
        else:
            self.weight_noise = 0

        self.action_noise = NormalActionNoise(mean=np.zeros(self.dof), sigma=self.noise_sigma * np.ones(self.dof))


        # Setup initial ProMP parameters
        self.contextual = bool(int(contextual))
        if self.contextual:
            self.context_hidden_layer = context_hidden_layer
            self.train = self.train_context
            self.collect_rollouts = self.collect_rollouts_context
        else:
            self.promp_params = initial_promp_params
            self.train = self.train_no_context
            self.collect_rollouts = self.collect_rollouts_no_context

        self.promp_policy_kwargs = promp_policy_kwargs

        # The initial reward of the best model
        self.best_model = -9000000

        self._setup_model()


    def _setup_model(self):
        """
        The function to initialize ProMP and critic network.
        """

        self._setup_lr_schedule() # learning rate schedule
        self._setup_critic_model() # initializing critic network
        self._convert_train_freq()

        '''
        if self.contextual:
            n_input = self.env.context().shape[0]
            assert len(self.env.context().shape) == 1, \
                "the contextual information should be reshape into one dimension."
            n_hidden = self.context_hidden_layer
            n_output = self.basis_num * self.dof
            self.actor_contextNN = ContextNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output).cuda()
            for i in self.actor_contextNN.parameters():
                i.data.fill_(0.01)

            self.actor_target_contextNN = ContextNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output).cuda()
            for i in self.actor_target_contextNN.parameters():
                i.data.fill_(0.01)
            #self.actor_contextNN = self.policy.actor #ContextNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output).cuda()
            #self.actor_target_contextNN = self.policy.actor_target #ContextNN(n_input=n_input, n_hidden=n_hidden, n_output=n_output).cuda()
        else:
            self.promp_params = self._setup_promp_params(self.promp_params)
        '''

        if self.contextual is False:
            self.promp_params = self._setup_promp_params(self.promp_params)

        # ProMP hyperparameters
        if 'controller_kwargs' in self.promp_policy_kwargs.keys():
            self.actor_kwargs = self.promp_policy_kwargs['controller_kwargs']
        else:
            self.actor_kwargs = None
        self.width = self.promp_policy_kwargs['width']
        self.weight_scale = self.promp_policy_kwargs['weight_scale']
        self.controller_type = self.promp_policy_kwargs['controller_type']
        self.zero_start = self.promp_policy_kwargs['zero_start']
        self.zero_basis = self.promp_policy_kwargs['zero_basis']

        self._setup_promp_model() # initializing ProMP

    def _setup_promp_params(self, initial_promp_params):
        """
        The function to build up ProMP initial weights into Tensor.
        If the input is int or float, ProMP parameters will be initialized
            by extending its size to self.basis_num * self.dof,
        and if the input is already a tensor, it will be checked whether its shape
            is same as self.basis_num * self.dof

        Input:
            initial_promp_params: the initial value of ProMP weights from users
        Return:
            the well-shaped ProMP weights which can be use in ProMP wrapper.
        """
        if initial_promp_params is None:
            initial_promp_params = 1 * th.ones(self.basis_num * self.dof)
        else:
            initial_promp_params = th.Tensor(np.array(initial_promp_params) * np.ones((self.basis_num, self.dof)))
        #else:
        #    if initial_promp_params.shape != (self.basis_num, self.dof):
        #        raise AssertionError(f'The shape of ProMP parameters should be {self.basis_num} * {self.dof}, '
        #                             f'now it is {initial_promp_params.shape}')
        #    initial_promp_params = th.Tensor(initial_promp_params)

        initial_promp_params = initial_promp_params.reshape(self.basis_num, self.dof)

        return (initial_promp_params.reshape(self.basis_num, self.dof)).to(device="cuda")


    def _setup_critic_model(self) -> None:
        """
        Initialize the critic model
        """
        self.set_random_seed(self.seed)
        if self.contextual:
            self.replay_buffer = ReplayBufferStep(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                context_space=self.env.context_space,
            )
        else:
            self.replay_buffer = ReplayBufferStep(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
            )
        self.policy_kwargs['basis_num'] = self.basis_num
        self.policy_kwargs['dof'] = self.dof
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            self.env.context_space,
            **self.policy_kwargs,  # pytype:disable=not-instantiable

        )
        self.policy = self.policy.to(self.device)
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target


    def _setup_promp_model(self) -> None:
        """
        Initialize the ProMP model
        """
        self.actor = DetPMPWrapper(self.env, num_dof=self.dof, num_basis=self.basis_num, width=self.width,
                                   controller_type=self.controller_type, weights_scale=self.weight_scale,
                                   zero_start=self.zero_start, zero_basis= self.zero_basis,
                                   step_length=self.max_episode_steps-self.pos_traj_steps,
                                   noise_sigma=self.noise_sigma,
                                   controller_kwargs=self.actor_kwargs,
                                   pos_traj_steps=self.pos_traj_steps, context=self.contextual)


        self.actor_target = DetPMPWrapper(self.env, num_dof=self.dof, num_basis=self.basis_num, width=self.width,
                                          controller_type=self.controller_type, weights_scale=self.weight_scale,
                                          zero_start=self.zero_start, zero_basis= self.zero_basis,
                                          step_length=self.max_episode_steps-self.pos_traj_steps,
                                          noise_sigma=self.noise_sigma,
                                          controller_kwargs=self.actor_kwargs,
                                          pos_traj_steps=self.pos_traj_steps,context=self.contextual
                                          )


        # Set the ProMP weights optimizer
        if self.contextual:
            self.env.reset()
            '''
            self.actor.mp.weights = self.actor_contextNN.forward(
                th.Tensor(self.env.context()).to(device='cuda')).reshape(self.basis_num, self.dof)
            self.actor_target.mp.weights = self.actor_target_contextNN.forward(
                th.Tensor(self.env.context()).to(device='cuda')).reshape(self.basis_num, self.dof)
            self.actor_optimizer = th.optim.Adam(self.actor_contextNN.parameters(), lr=self.actor_learning_rate)
            '''
            self.actor.mp.weights = self.policy.actor(th.Tensor(
                self.env.context()).to(device='cuda').reshape(1,-1)).reshape(self.basis_num, self.dof)
            self.actor_target.mp.weights = self.policy.actor_target(th.Tensor(
                self.env.context()).to(device='cuda').reshape(1,-1)).reshape(self.basis_num, self.dof)
            self.actor_optimizer = self.policy.actor.optimizer
            self.actor_optimizer.param_groups[0]['lr'] = self.actor_learning_rate
            #self.weights_optimizer = th.optim.Adam([self.actor.mp.pos_features], lr=self.actor_learning_rate)
        else:
            # Pass the promp parameters value to ProMP weights
            self.actor.mp.weights = self.promp_params.to(device='cuda')
            (self.actor.mp.weights).requires_grad = True  # Enable the gradient of ProMP weights
            self.actor_optimizer = th.optim.Adam([self.actor.mp.weights], lr=self.actor_learning_rate)

            # Set target ProMP weights by target delay
            self.actor_target.mp.weights = self.actor.mp.weights * self.tau
            self.actor_target.mp.weights = self.actor_target.mp.weights.to(device='cuda')

        # Update the reference trajectory according to weights
        self.actor.update()
        self.actor_target.update()


    def train_no_context(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        The function updates the parameters of critic network and ProMP weights.
        """

        # evaluate the current policy, and save the reward and the episode length
        self.actor.update()
        if self.num_timesteps/10000 >= self.eval_numsteps:
            self.eval_reward, eval_epi_length = self.actor.eval_rollout(self.env)
            self.env.reset()

        # learning rate and noise schedule
        if self.need_schedule:
            self.schedule.schedule(model=self)

        # save the best evaluate reward
        if self.best_model < self.eval_reward:
            self.best_model = self.eval_reward
            np.savez(self.data_path + "/best_model", self.actor.mp.weights.cpu().detach().numpy())

        # save current policy parameters
        np.savez(self.data_path + "/algo_mean", self.actor.mp.weights.cpu().detach().numpy())
        np.savez(self.data_path + "/pos_features", self.actor.mp.pos_features.cpu().detach().numpy())
        np.savez(self.data_path + "/vel_features", self.actor.mp.vel_features.cpu().detach().numpy())

        # critic learning rate schedule
        self._update_learning_rate([self.critic.optimizer])

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
                #self.weights_Contextoptimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                #self.controller_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                #self.weights_optimizer.step()
                #self.controller_optimizer.step()

                # Update actor target
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                self.actor_target.mp.weights = (self.actor.mp.weights * self.tau
                                                    + (1 - self.tau) * self.actor_target.mp.weights).to(device="cuda")

                # update the reference trajectory in ProMP
                self.actor.update()
                self.actor_target.update()

        # supervise the trajectory and weights, should be deleted when finished
        print("context", self.env.context())
        print("weights", self.actor.mp.weights[-1])

        #for i in self.actor_contextNN.parameters():
        #    print(i[0][:10])


        # tensorboard logger
        if self.num_timesteps/10000 >= self.eval_numsteps:

            logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                logger.record("train/actor_loss", np.mean(actor_losses))
            logger.record("train/critic_loss", np.mean(critic_losses))
            logger.record("eval/noise_reward", np.mean(np.array(self.reward_with_noise)))
            logger.record("train/actor_learning_rate", self.actor_learning_rate)
            logger.record("train/gradient_steps", gradient_steps)
            logger.record("train/noise_sigma_action", self.noise_sigma)
            logger.record("train/noise_sigma_weights", self.weight_noise)
            logger.record("train/num_basis", self.basis_num)
            logger.record("eval/mean_reward", self.eval_reward)
            logger.record("eval/episode_length", eval_epi_length)
            self.eval_numsteps +=1
            if "Meta" in str(self.env):
                logger.record("eval/last_success", self.actor.last_success)
                logger.record("eval/last_object_to_target", self.actor.last_target_object)
                logger.record("eval/min_object_to_target", self.actor.min_target_object)
                logger.record("eval/control_cost", self.actor.control_cost)
            elif "Hopper" in str(self.env):
                logger.record("eval/max_height", self.actor.max_height)
                logger.record("eval/min_goal_dist", self.actor.min_goal_dist)
            elif "DeepMind" in str(self.env):
                logger.record("eval/success", self.actor.last_success)


    def learn(self, total_timesteps: int, callback: MaybeCallback = None, log_interval: int = 4,
              eval_env: Optional[GymEnv] = None, eval_freq: int = -1, n_eval_episodes: int = 5,
              tb_log_name: str = "run", eval_log_path: Optional[str] = None, reset_num_timesteps: bool = True,
              ) -> "OffPolicy":
        """
        This function begins the procedure of the whole learning proces.
        """
        total_timesteps, callback = self._setup_learn(total_timesteps, eval_env, callback, eval_freq,
                                                      n_eval_episodes, eval_log_path, reset_num_timesteps,
                                                      tb_log_name)

        callback.on_training_start(locals(), globals())

        # start learning process
        while self.num_timesteps < total_timesteps:
            rollout = self.collect_rollouts(
                self.env,
                train_freq=self.train_freq,
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


    def _sample_action(self, episode_timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        This function samples the data from the environment.
        """
        if self.num_timesteps < self.learning_starts:
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            unscaled_action = self.actor.get_action(episode_timesteps)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            if self.contextual:
                #unscaled_action = np.tanh(unscaled_action)
                action = unscaled_action
                if self.weight_noise_judge is False:
                    unscaled_action += self.action_noise().reshape(-1)
                buffer_action = action
                '''
                scaled_action = self.policy.scale_action(unscaled_action).reshape(1, self.dof)
                scaled_action = np.clip(scaled_action, -1, 1)
                if self.weight_noise_judge is False:
                    scaled_action += self.action_noise().reshape(-1)
                buffer_action = scaled_action
                action = self.policy.unscale_action(scaled_action)
                '''
            else:
                #unscaled_action = np.tanh(unscaled_action)
                if self.weight_noise_judge is False:
                    unscaled_action += self.action_noise().reshape(-1)
                action = unscaled_action
                buffer_action = action
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = np.clip(unscaled_action, -1, 1)
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

    def _store_transition(self, replay_buffer: ReplayBufferStep, buffer_action: np.ndarray,
                          new_obs: np.ndarray, reward: np.ndarray, done: np.ndarray,
                          infos: List[Dict[str, Any]], steps: np.ndarray, next_steps: np.ndarray,
                          context: np.ndarray = None) -> None:
        """
        Store the transitions into Replay Buffer.
        """
        self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward
        next_obs = new_obs_
        steps = np.array(steps).reshape(-1)
        next_steps = np.array(next_steps).reshape(-1)
        context = np.array(context).reshape(-1)

        replay_buffer.add(self._last_original_obs, next_obs, buffer_action, reward_, done, steps, next_steps, context)

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_

    def collect_rollouts_no_context(self, env: VecEnv, callback: BaseCallback, train_freq: TrainFreq,
                         replay_buffer: ReplayBufferStep, learning_starts: int = 0,
                         log_interval: Optional[int] = None,
                         ) -> RolloutReturn:

        """
        Collect the data.
        """

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0


        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        #  the timestep information in each episode
        if self.episode_timesteps == 0:
            self.reward_with_noise = []
            done = False
            #self.weights = self.actor.mp.weights
            #print("rollout, weight", self.actor.mp.weights)
            #self.actor.mp.weights = noise_weights #th.Tensor(self.weight_noise()).to(device='cuda')
            self.actor.update()
            if self.weight_noise_judge:
                self.actor.trajectory_np += self.actor.mp.pos_features_np \
                                            @ self.param_noise().reshape(self.basis_num, self.dof)
                self.actor.velocity_np += self.actor.mp.vel_features_np \
                                          @ self.param_noise().reshape(self.basis_num, self.dof)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes) or \
                self.ls_number < self.learning_starts:

            # loop for episode
            done = False
            episode_reward = 0.0
            self.obs = []
            self.actions = []


            while not done:  # loop for steps during one episode (timesteps plus one)

                # Select action according to policy
                action, buffer_action = self._sample_action(self.episode_timesteps)

                # Rescale and perform action
                if 'Meta' in str(env):
                    action = action.reshape(-1)

                new_obs, reward, done, infos = env.step(action)
                new_obs = np.hstack([new_obs]).reshape(1, -1)
                action = action.reshape(1,-1)
                #action = action + self.noise().reshape(action.shape)
                #action = action.reshape(1, -1)
                new_obs = new_obs.reshape(1,-1)
                self.actions.append(action)

                self.num_timesteps += 1
                self.episode_timesteps += 1
                self.ls_number += 1
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                episode_reward += reward

                # Store data in replay buffer (normalized action and unnormalized observation)
                next_step = self.episode_timesteps
                if self.episode_timesteps == self.max_episode_steps:
                    next_step = self.max_episode_steps-1
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos,
                                       self.episode_timesteps - 1, next_step)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # should I use this?
                # if the environment ends before one episode length, restart the environment
                #if not (should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes) and
                #        self.ls_number < self.learning_starts):
                #    env.reset()
                #    self.episode_timesteps = 0
                #    break

                #env.render()

            if done:
                # save the reward of the noisy sampling environment
                if hasattr(self.env, "rewards_no_ip"):
                    self.reward_with_noise.append(self.env.rewards_no_ip)  # the total reward without initial phase
                else:
                    self.reward_with_noise.append(episode_reward)
                env.reset()
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(self.episode_timesteps)
                self.episode_timesteps = 0

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    print("")
                    self._dump_logs()

                self.actor.update()
                if self.weight_noise_judge:
                    self.actor.trajectory_np += self.actor.mp.pos_features_np \
                                                @ self.param_noise().reshape(self.basis_num, self.dof)

                    self.actor.velocity_np += self.actor.mp.vel_features_np \
                                              @ self.param_noise().reshape(self.basis_num, self.dof)

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)


    def _convert_train_freq(self) -> None:
        """
        This function builds up the training frequency.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")
            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(
                    f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        This function saves the replay buffer.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)
    def _setup_learn(self, total_timesteps: int, eval_env: Optional[GymEnv], callback: MaybeCallback = None,
                     eval_freq: int = 10000, n_eval_episodes: int = 5, log_path: Optional[str] = None,
                     reset_num_timesteps: bool = True, tb_log_name: str = "run",
                     ) -> Tuple[int, BaseCallback]:
        """
        This function setups the learning process.
        """
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

    def update_context(self, context):
        self.actor.mp.weights = self.policy.actor.forward(
            context.reshape(1,-1)).reshape(self.basis_num, self.dof)
        self.actor_target.mp.weights = self.policy.actor_target.forward(
           context.reshape(1,-1)).reshape(self.basis_num, self.dof)
        self.actor.update()
        self.actor_target.update()


    def update_context_in_training(self, context, steps):
        self.actor.mp.weights = self.policy.actor.forward(
            context).reshape(self.batch_size, self.basis_num, self.dof)
        self.actor_target.mp.weights = self.policy.actor_target.forward(
           context).reshape(self.batch_size, self.basis_num, self.dof)
        self.actor.update_context(steps)
        self.actor_target.update_context(steps)



    def train_context(self, gradient_steps: int, batch_size: int = 100) -> None:
        """
        The function updates the parameters of critic network and ProMP weights.
        """

        # evaluate the current policy, and save the reward and the episode length
        if self.num_timesteps/10000 >= self.eval_numsteps:
            self.eval_reward = 0
            eval_epi_length = 0
            if "Hopper" in str(self.env):
                self.max_height = 0
                self.min_goal_dist = 0
            elif "Meta" in str(self.env):
                self.last_success = 0
                self.success_rate = []
                self.last_target_object = 0
                self.min_target_object = 0
            elif "DeepMind" in str(self.env):
                self.success = []
            self.eval_num = 10
            for i in range(self.eval_num):
                self.update_context(th.Tensor(self.env.context()).to(device='cuda'))
                episode_reward, eval_length = self.actor.eval_rollout(self.env)
                self.env.reset()
                self.eval_reward += episode_reward
                eval_epi_length += eval_length
                if "Hopper" in str(self.env):
                    self.max_height += self.actor.max_height
                    self.min_goal_dist += self.actor.min_goal_dist
                elif "Meta" in str(self.env):
                    self.success_rate.append(self.actor.success_rate)
                    self.last_success += self.actor.last_success
                    self.last_target_object += self.actor.last_target_object
                    self.min_target_object += self.actor.min_target_object
                elif "DeepMind" in str(self.env):
                    self.success.append(self.actor.last_success)
            if "Hopper" in str(self.env):
                self.max_height /= self.eval_num
                self.min_goal_dist /= self.eval_num
            elif "Meta" in str(self.env):
                self.success_rate = np.mean(np.array(self.success_rate))
                self.last_success /= self.eval_num
                self.last_target_object /= self.eval_num
                self.min_target_object /= self.eval_num
            elif "DeepMind" in str(self.env):
                self.success_rate = np.mean(np.array(self.success))
            self.eval_reward /= self.eval_num
            eval_epi_length /= self.eval_num

            print("episode_reward", self.eval_reward)

        # learning rate and noise schedule
        if self.need_schedule:
            self.schedule.schedule(model=self)

        # save the best evaluate reward
        if self.best_model < self.eval_reward:
            self.best_model = self.eval_reward
            #np.savez(self.data_path + "/best_model", self.actor.mp.weights.cpu().detach().numpy())
            th.save(self.policy.actor.state_dict(), self.data_path + '/best_model.pt')

        # save current policy parameters
        #np.savez(self.data_path + "/algo_mean", self.actor.mp.weights.cpu().detach().numpy())
        #np.savez(self.data_path + "/pos_features", self.actor.mp.pos_features.cpu().detach().numpy())
        #np.savez(self.data_path + "/vel_features", self.actor.mp.vel_features.cpu().detach().numpy())
        th.save(self.policy.actor.state_dict(), self.data_path + '/algo_mean.pt')


        # critic learning rate schedule
        self._update_learning_rate([self.critic.optimizer])

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

                self.update_context_in_training(replay_data.context, replay_data.next_steps)
                #self.env.reset()

                next_actions = (self.actor_target.predict_action_context(next_step, replay_data.next_observations) + noise).clamp(-1, 1)

                # Compute the next Q-values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions, (next_step+1)/self.max_episode_steps), dim=1)
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

                self.update_context_in_training(replay_data.context, replay_data.steps)
                #self.env.reset()

                act = self.actor.predict_action_context(replay_data.steps, replay_data.observations)
                actor_loss = -self.critic.q1_forward(replay_data.observations, act,  (replay_data.steps+1)/self.max_episode_steps).mean()
                actor_losses.append(actor_loss.item())

                # Optimize the actor
                self.actor_optimizer.zero_grad()
                #self.controller_optimizer.zero_grad()
                actor_loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), 0.5)
                self.actor_optimizer.step()
                #th.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                #self.controller_optimizer.step()

                # Update actor target
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.policy.actor.parameters(), self.policy.actor_target.parameters(), self.tau)
                #self.actor_target.mp.pos_features = (self.actor.mp.pos_features * self.tau
                #                        + (1 - self.tau) * self.actor_target.mp.pos_features).to(device="cuda")
                #self.update_context()
                #self.env.reset()

        self.update_context(th.Tensor(self.env.context()).to(device='cuda'))

        # supervise the trajectory and weights, should be deleted when finished
        print("context", self.env.context())
        print("weights", self.actor.mp.weights[0])
        #print("features", self.actor.mp.pos_features[0])
        #for i in self.actor_contextNN.parameters():
        #    print(i[0][:10])


        # tensorboard logger
        if self.num_timesteps/10000 >= self.eval_numsteps:
            logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            if len(actor_losses) > 0:
                logger.record("train/actor_loss", np.mean(actor_losses))
            logger.record("train/critic_loss", np.mean(critic_losses))
            logger.record("eval/noise_reward", np.mean(np.array(self.reward_with_noise)))
            logger.record("train/actor_learning_rate", self.actor_learning_rate)
            logger.record("train/gradient_steps", gradient_steps)
            logger.record("train/noise_sigma_action", self.noise_sigma)
            logger.record("train/noise_sigma_weights", self.weight_noise)
            logger.record("train/num_basis", self.basis_num)
            logger.record("eval/mean_reward", self.eval_reward)
            logger.record("eval/episode_length", eval_epi_length)
            self.eval_numsteps += 1
            if "Meta" in str(self.env):
                logger.record("eval/last_success", self.last_success)
                logger.record("eval/success_rate", self.success_rate)
                logger.record("eval/last_object_to_target", self.last_target_object)
                logger.record("eval/min_object_to_target", self.min_target_object)
                logger.record("eval/control_cost", self.actor.control_cost)
            elif "Hopper" in str(self.env):
                logger.record("eval/max_height", self.max_height)
                logger.record("eval/min_goal_dist", self.min_goal_dist)
            elif "DeepMind" in str(self.env):
                self.success = []
                logger.record("eval/success", self.success_rate)

    def collect_rollouts_context(self, env: VecEnv, callback: BaseCallback, train_freq: TrainFreq,
                         replay_buffer: ReplayBufferStep, learning_starts: int = 0,
                         log_interval: Optional[int] = None,
                         ) -> RolloutReturn:

        """
        Collect the data.
        """

        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0


        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True

        #  the timestep information in each episode
        if self.episode_timesteps == 0:
            done = False
            #self.actor.update()
            self.update_context(th.Tensor(self.env.context()).to(device='cuda'))
            if self.weight_noise_judge:
                self.actor.trajectory_np += self.actor.mp.pos_features_np \
                                            @ self.param_noise().reshape(self.basis_num, self.dof)

                self.actor.velocity_np += self.actor.mp.vel_features_np \
                                          @ self.param_noise().reshape(self.basis_num, self.dof)


        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes) or \
                self.ls_number < self.learning_starts:

            # loop for episode
            done = False
            episode_reward = 0.0
            self.obs = []
            self.actions = []


            while not done:  # loop for steps during one episode (timesteps plus one)

                # Select action according to policy
                action, buffer_action = self._sample_action(self.episode_timesteps)

                if 'Meta' in str(env):
                    action = action.reshape(-1)

                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)

                context = self.env.context().reshape(1,-1)
                new_obs = np.hstack([new_obs]).reshape(1, -1)
                action = action.reshape(1,-1)
                new_obs = new_obs.reshape(1,-1)
                self.actions.append(action)

                self.num_timesteps += 1
                self.episode_timesteps += 1
                self.ls_number += 1
                self.obs.append(new_obs)
                num_collected_steps += 1

                # Give access to local variables
                callback.update_locals(locals())
                episode_reward += reward

                # Store data in replay buffer (normalized action and unnormalized observation)
                next_step = self.episode_timesteps
                if self.episode_timesteps == self.max_episode_steps:
                    next_step = self.max_episode_steps-1

                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos,
                                       self.episode_timesteps - 1, next_step, context)

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # should I use this?
                # if the environment ends before one episode length, restart the environment
                #if not (should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes) and
                #        self.ls_number < self.learning_starts):
                #    env.reset()
                #    self.episode_timesteps = 0
                #    break

                #env.render()

            if done:
                # save the reward of the noisy sampling environment
                if hasattr(self.env, "rewards_no_ip"):
                    self.reward_with_noise = self.env.rewards_no_ip  # the total reward without initial phase
                else:
                    self.reward_with_noise = episode_reward
                env.reset()
                num_collected_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(self.episode_timesteps)
                self.episode_timesteps = 0

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

                self.update_context(th.Tensor(self.env.context()).to(device='cuda'))

                if self.weight_noise_judge:
                    self.actor.trajectory_np += self.actor.mp.pos_features_np \
                                                @ self.param_noise().reshape(self.basis_num, self.dof)

                    self.actor.velocity_np += self.actor.mp.vel_features_np \
                                              @ self.param_noise().reshape(self.basis_num, self.dof)

        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)


    def render_rollout(self, weights, env, pos, vel):
        """
        This function render the environment.

        Input:
            weights: the learned weights.
            env: the environment we want to render.
        """
        import time
        self.actor.mp.weights = th.Tensor(weights).to(device='cuda').reshape(-1,self.dof)
        #self.actor.mp.weights = th.Tensor(np.array([-0.1,-0.1,-0.1,-0.1])*np.ones((5,4))).to(device='cuda')
        self.actor.update()
        #print("pos_model",self.actor.mp.pos_features_np)


        ob1 = []
        ob = env.reset()
        if self.contextual:
            self.update_context(th.Tensor(self.env.context()).to(device='cuda'))

        print("algorithm", self.actor.mp.weights)
        jump_height = []
        goal_dist =[]

        for i in range(1):
            if i == 1:
                noise_dist = NormalActionNoise(mean=np.zeros(self.dof), sigma=[0] * np.ones(self.dof))
            else:
                noise_dist = NormalActionNoise(mean=np.zeros(self.dof),
                                               sigma=0 * np.ones(self.dof))
            rewards = 0
            step_length = self.max_episode_steps
            #print("ob", ob, self.env.sim.data.qpos, self.env.sim.data.qvel)
            obs = []
            ac1 = []
            infos = []
            infos = []
            import time
            if "Meta" in str(env):
                for i in range(int(self.max_episode_steps)):
                    time.sleep(1)
                    ac = self.actor.get_action(i)
                    #if self.contextual:
                    #    ac = np.tanh(ac)
                    #if self.pos_traj_steps < i:
                    #    ac[:3] = 0#self.env.sim.data.mocap_pos
                    #ac = np.tanh(ac)
                    acs = np.clip(ac, -1, 1).reshape(self.dof) + noise_dist()
                    #acs[-1] = 1
                    #acs[-1] = .5
                    #if i > 50:
                    #    acs[-1] = -0.1
                    #acs[2] = 0

                    ob, reward, dones, info = env.step(acs)
                    print(i, acs, reward,self.actor.trajectory_np[i], info['success'])
                    infos.append(info['obj_to_target'])
                    obs.append(self.env.sim.data.mocap_pos.copy())
                    #if i == 59 or i == 89 or i == 199 or i == 19 or i == 1:
                    #if i == 199:
                    #    time.sleep(5)
                    rewards += reward
                    env.render(False)
                ob1 = ob

                infos = np.array(infos)
                print(np.min(infos))
                print("rewards", rewards)
            else:
                import time
                for i in range(step_length):
                    #print(self.actor.mp.weights)
                    #time.sleep(0.01)
                    ac = self.actor.get_action(i, noise=0)
                    #if self.contextual:
                    #    ac = np.tanh(ac)
                    print(i,ac)
                    ac = np.clip(ac, -1, 1).reshape(1, self.dof)
                    obs, reward, done, info = env.step(ac)
                    ob1.append(obs)
                    time.sleep(0.1)
                    #rewards += info["reward"]
                    #jump_height.append(info["height"])
                    #goal_dist.append(info["goal_dist"])
                    rewards += reward
                    env.render()
                    #if i == 0 or i == 125 or i == 249:
                    #    time.sleep(5)
                    if done:
                        step_length = i + 1
                        break
                print("context", self.env.context())
                print("rewards", rewards)


            env.render()
            #if i == 0 or i==125 or i==249:
            #    time.sleep(5)
            env.close()

            import matplotlib.pyplot as plt
            jump_height = np.array(jump_height)
            goal_dist = np.array(goal_dist)
            np.savez("jump_hieght"+"Epi.TD3",jump_height)
            np.savez("goal_dist"+"Epi.TD3",goal_dist)
            # plt.plot(goal[:, 0], label='x axis')
            plt.plot(jump_height, label='TD3')
            # plt.plot(goal[:, 1], label='y axis')
            #plt.plot(goal[:, 1], label='y-axis')
            # plt.plot(goal[:, 2], label='x axis')
            plt.legend()
            plt.xlabel("timesteps")
            plt.ylabel("jump height")
            # plt.show()
            # plt.title('Goal Position')
            import tikzplotlib
            tikzplotlib.save("hopper_height.tex")
            #plt.show()
            print(reward)