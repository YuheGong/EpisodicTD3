from abc import ABC
import gym
import numpy as np
from .detpmp_model import DeterministicProMP
import torch as th
from .controller import PosController, PDController, VelController, PIDController, MetaWorldController
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


class DetPMPWrapper(ABC):
    """
    This class is the wrapper for Probabilistic Movement Primitives model.

    :param env: for any regular OpenAI Gym environment.
    :param num_dof: the degree of freedom of the robot
    :param num_basis: the number of Gaussian Basis Functions.
    :param width: the width of Gaussian Basis Functions.
    :param step_length: the episode length.
    :param weights_scale: the scale of the weights
    :param zero_start: whether start from the initial position or not
    :param zero_basis: the basis functions when zero_start is True.
    :param mp_kwargs: the parameter of the controller
    """

    def __init__(
            self,
            env: gym.Wrapper,
            num_dof: int,
            num_basis: int,
            width: int,
            step_length=None,
            weights_scale=1,
            zero_start=False,
            zero_basis=0,
            noise_sigma=0,
            pos_traj_steps = 0,
            context = False,
            **controller_kwargs):

        self.controller_type = controller_kwargs['controller_type']
        self.controller_setup(env=env, controller_kwargs=controller_kwargs, num_dof=num_dof)

        self.zero_start = zero_start
        self.context = context
        self.start_traj = None
        self.trajectory = None
        self.velocity = None
        self.pos_traj_steps = pos_traj_steps
        #self.noise = NormalActionNoise(mean=np.zeros(num_dof * num_basis), sigma=noise_sigma * np.ones(num_dof * num_basis))

        self.step_length = step_length
        self.max_episodes = self.step_length + pos_traj_steps
        self.env = env
        dt = self.env.dt


        self.num_dof = num_dof
        self.mp = DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width,
                                     zero_start=zero_start, n_zero_bases=zero_basis,
                                     step_length=self.step_length, dt=dt, weight_scale=weights_scale,
                                     pos_traj_steps=pos_traj_steps)


    def controller_setup(self, env, controller_kwargs, num_dof):
        """
        This function builds up the controller of ProMP.
        """
        if self.controller_type == 'motor':
            self.controller = PDController(env, p_gains=controller_kwargs['controller_kwargs']['p_gains'],
                                           d_gains=controller_kwargs['controller_kwargs']['d_gains'], num_dof=num_dof)
        elif self.controller_type == 'position':
            self.controller = PosController(env, num_dof=num_dof)
        elif self.controller_type == 'velocity':
            self.controller = VelController(env,p_gains=controller_kwargs['controller_kwargs']['p_gains'],
                                           d_gains=controller_kwargs['controller_kwargs']['d_gains'], num_dof=num_dof)
        elif self.controller_type == 'pid':
            self.controller = PIDController(env, p_gains=controller_kwargs['controller_kwargs']['p_gains'],
                                            i_gains=controller_kwargs['controller_kwargs']['i_gains'],
                                            d_gains=controller_kwargs['controller_kwargs']['d_gains'],
                                            num_dof=num_dof)
        elif self.controller_type == 'MetaWorld':
            self.controller = MetaWorldController(env, num_dof=num_dof)
        else:
            raise AssertionError("controller not exist")

    def update(self):
        """
        This function build up the reference trajectory of ProMP
        according to the current weights in each iteration.
        """
        # torch version of the reference trajectory
        _,  self.trajectory, self.velocity, __ = self.mp.compute_trajectory()
        # add initial position
        if self.zero_start:
            if self.controller_type == 'motor':
                self.trajectory += th.Tensor(
                    self.env.current_pos.reshape(self.num_dof)).to(device='cuda')
            elif self.controller_type == 'position':
                self.trajectory += th.Tensor(self.env.current_pos().reshape(self.num_dof)).to(
                    device='cuda')
            elif self.controller_type == "MetaWorld":
                self.trajectory[:,:-1] += th.Tensor(self.env.current_pos()[:-1].reshape(self.num_dof-1)).to(
                    device='cuda')
            elif self.controller_type == 'pid':
                #self.trajectory += th.Tensor(
                #    self.controller.obs()[-3 * self.num_dof:-2 * self.num_dof].reshape(self.num_dof)).to(device='cuda')
                self.trajectory += th.Tensor(
                    self.env.current_pos.reshape(self.num_dof)).to(device='cuda')

        # numpy version of the reference trajectory
        self.trajectory_np = self.trajectory.cpu().detach().numpy()
        self.velocity_np = self.velocity.cpu().detach().numpy()
        #self.acceleration_np = self.acceleration.cpu().detach().numpy()

    def update_context(self, steps):
        """
        This function build up the reference trajectory of ProMP
        according to the current weights in each iteration.
        """
        # torch version of the reference trajectory
        self.trajectory = self.mp.pos_features[steps] @ self.mp.weights
        self.velocity = self.mp.vel_features[steps] @ self.mp.weights
        #self.acceleration = self.mp.acc_features[steps] @ self.mp.weights

        # add initial position
        if self.zero_start:
            if self.controller_type == 'motor':
                self.trajectory += th.Tensor(self.env.current_pos.reshape(self.num_dof)).to(device='cuda')
            elif self.controller_type == 'position':
                self.trajectory += th.Tensor(self.env.current_pos().reshape(self.num_dof)).to(
                    device='cuda')
            elif self.controller_type == "MetaWorld":
                self.trajectory[:,:, :] += th.Tensor(self.env.current_pos()[:].reshape(self.num_dof)).to(
                    device='cuda')
            elif self.controller_type == 'pid':
               self.trajectory += th.Tensor(
                    self.env.current_pos.reshape(self.num_dof)).to(device='cuda')

        # numpy version of the reference trajectory
        #self.trajectory_np = self.trajectory.cpu().detach().numpy()
        #self.velocity_np = self.velocity.cpu().detach().numpy()
        #self.acceleration_np = self.acceleration.cpu().detach().numpy()

    def predict_action_context(self, step, observation):
        """
        This function predicts the actions according to the Replay Buffer observations.
        It is used for critic network and actor policy updating.

        Input:
            step: the timestep information stored in Replay Buffer.
            observation: the observation stored in Replay Buffer.
        Return:
            action: the action based on current ProMP parameters.
        """
        self.positions = self.trajectory.reshape(-1, self.num_dof)
        self.velocities = self.velocity.reshape(-1, self.num_dof)
        #self.accelerations = self.acceleration.reshape(-1, self.num_dof)
        actions = self.controller.predict_actions(self.positions, self.velocities, None, observation)
        #actions = th.tanh(actions)
        return actions

    def predict_action(self, step, observation):
        """
        This function predicts the actions according to the Replay Buffer observations.
        It is used for critic network and actor policy updating.

        Input:
            step: the timestep information stored in Replay Buffer.
            observation: the observation stored in Replay Buffer.
        Return:
            action: the action based on current ProMP parameters.
        """
        self.positions = self.trajectory[step].reshape(-1, self.num_dof)
        self.velocities = self.velocity[step].reshape(-1, self.num_dof)
        #self.accelerations = self.acceleration[step].reshape(-1, self.num_dof)
        actions = self.controller.predict_actions(self.positions, self.velocities, None, observation)
        #actions = th.tanh(actions)
        return actions

    def get_action(self, timesteps, noise=0):
        """
        This function generates the actions according to the observation of the environment.
        It is used for interacting with the environment.

        Input:
            step: the timestep information.
        Return:
            action: the action used for indicating the movements of the robot.
        """

        trajectory = self.trajectory_np[timesteps].copy() #+ noise_traj
        velocity = self.velocity_np[timesteps].copy()
        #acceleration = self.acceleration_np[timesteps].copy()
        action, des_pos, des_vel = self.controller.get_action(trajectory, velocity, None)
        #action = np.tanh(action)
        return action

    def eval_rollout(self, env):
        """
        This function evaluate the current ProMP.

        Input:
            step: the environment without normalization.
            (We don't use normalization for the environment in our implementation,
            so this environment is same as the environment we used for sampling data.)
        Return:
            episode_reward: the reward of one episode based on current ProMP model.
            step_length: the step length of one episode based on current ProMP model.
        """
        rewards = 0
        step_length = self.max_episodes
        #env.reset()
        if "Meta" in str(env):
            self.min_target_object = 100
            self.last_target_object = 0
            self.last_success = 0
            self.control_cost = 0
            self.success_rate = []
            for i in range(int(self.max_episodes)):
                ac = self.get_action(i)
                #if self.context:
                #    ac = np.tanh(ac)
                ac = np.clip(ac, -1, 1).reshape(self.num_dof)
                obs, reward, dones, info = env.step(ac)
                rewards += reward
                self.control_cost += np.sum(np.square(ac))
                self.success_rate.append(info['success'])
                if self.min_target_object > info['obj_to_target']:
                    self.min_target_object = info['obj_to_target']
            self.success_rate = np.any(np.array(self.success_rate))
            self.last_success = info['success']
            self.last_target_object = info['obj_to_target']
        else:
            import time
            for i in range(self.max_episodes):
                ac = self.get_action(i, noise=0)
                #if self.context:
                #    ac = np.tanh(ac)
                ac = np.clip(ac, -1, 1).reshape(1,self.num_dof)
                obs, reward, done, info = env.step(ac)
                rewards += reward
                if done:
                    step_length = i + 1
                    break
            if "Hopper" in str(env):
                self.max_height = info["max_height"]
                self.min_goal_dist = info["min_goal_dist"]
            elif "DeepMind" in str(env):
                self.last_success = info['success']

        if hasattr(self.env, "rewards_no_ip"):
            episode_reward = env.rewards_no_ip  # the total reward without initial phase
        else:
            episode_reward = rewards


        return episode_reward, step_length

    # should be deleted when finished, use render_rollout to render the environment
    '''
    def load(self, action):
        action = torch.FloatTensor(action)
        params = action.reshape(self.mp.n_basis, self.mp.n_dof) * self.weights_scale
        self.mp.weights = params.to(device="cuda")
        _, des_pos, des_vel, __ = self.mp.compute_trajectory(self.mp.weights)
        des_pos += th.Tensor(self.controller.obs()[-2*self.num_dof:-self.num_dof]).to(device='cuda')
        return des_pos, des_vel
    '''

