from abc import ABC
import torch
import gym
import numpy as np
from .detpmp_model import DeterministicProMP
import torch as th
from stable_baselines3.common.noise import NormalActionNoise
from .controller import PosController, PDController, VelController


class DetPMPWrapper(ABC):

    def __init__(self, env: gym.Wrapper, num_dof: int, num_basis: int, width: int, step_length=None,
                 weights_scale=1, zero_start=False, zero_goal=False, noise_sigma=None,
                 **mp_kwargs):

        self.policy_type = mp_kwargs['policy_type']
        self.zero_start = zero_start
        self.controller_setup(env=env, policy_kwargs=mp_kwargs, num_dof=num_dof)

        self.weights_scale = weights_scale
        self.trajectory = None
        self.velocity = None

        self.step_length = step_length
        self.env = env
        dt = self.env.dt

        # set the exploration noise for reference trajectory
        self.noise_sigma = noise_sigma
        n_actions = (num_dof,)
        self.noise_traj = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.noise_sigma * np.ones(n_actions))

        self.num_dof = num_dof
        self.mp = DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width,
                                     zero_start=zero_start, n_zero_bases=2,
                                     step_length=self.step_length, dt=dt)

    def controller_setup(self, env, policy_kwargs, num_dof):
        if self.policy_type == 'motor':
            self.controller = PDController(env, p_gains=policy_kwargs['policy_kwargs']['p_gains'],
                                           d_gains=policy_kwargs['policy_kwargs']['d_gains'], num_dof=num_dof)
        elif self.policy_type == 'position':
            self.controller = PosController(env, num_dof=num_dof)
        elif self.policy_type == 'velocity':
            self.controller = VelController(env, num_dof=num_dof)
        else:
            raise AssertionError("controller not exist")

    def predict_action(self, step, observation):
        self.calculate_traj = self.trajectory[step].reshape(-1, self.num_dof)
        self.calculate_vel = self.velocity[step].reshape(-1, self.num_dof)
        actions = self.controller.predict_actions(self.calculate_traj, self.calculate_vel, observation)
        return actions

    def update(self):
        weights = self.mp.weights * self.weights_scale
        _,  self.trajectory, self.velocity, __ = self.mp.compute_trajectory(weights)

        if self.zero_start:
            if self.policy_type == 'motor':
                self.trajectory += th.Tensor(
                    self.controller.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)).to(device='cuda')
            elif self.policy_type == 'position':
                self.trajectory += th.Tensor(self.controller.obs()[-self.num_dof:].reshape(self.num_dof)).to(device='cuda')

        self.trajectory_np = self.trajectory.cpu().detach().numpy()
        self.velocity_np = self.velocity.cpu().detach().numpy()


    def get_action(self, timesteps):
        """
        This function generates the actions through the controller
            according to the reference trajectory and reference velocity.
        """

        trajectory = self.trajectory_np[timesteps]
        velocity = self.velocity_np[timesteps] + self.noise_traj()

        action, des_pos, des_vel = self.controller.get_action(trajectory, velocity)
        return action


    def eval_rollout(self, env, a):
        rewards = 0
        for t, pos_vel in enumerate(zip(self.trajectory_np, self.velocity_np)):
            des_pos = pos_vel[0]
            des_vel = pos_vel[1]
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            ac = np.clip(ac, -1, 1).reshape(1,self.num_dof)
            obs, reward, done, info = env.step(ac)
            rewards += reward
        return env.rewards_no_ip


    def load(self, action):
        action = torch.FloatTensor(action)
        params = action.reshape(self.mp.n_basis, self.mp.n_dof) * self.weights_scale
        self.mp.weights = params.to(device="cuda")
        _, des_pos, des_vel, __ = self.mp.compute_trajectory(self.mp.weights)
        des_pos += th.Tensor(self.controller.obs()[-2*self.num_dof:-self.num_dof]).to(device='cuda')
        return des_pos, des_vel

    def render_rollout(self, action, env, noise,  pos_feature, vel_feature):
        import time
        env.reset()

        _, self.trajectory, self.velocity, __ = self.mp.compute_trajectory(th.Tensor(action).to(device='cuda'))

        #if self.zero_start:
        #    if self.policy_type == 'motor':
        #        self.trajectory += th.Tensor(
        #            self.controller.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)).to(device='cuda')
        #    elif self.policy_type == 'position':
        #        self.trajectory += th.Tensor(self.controller.obs()[-self.num_dof:].reshape(self.num_dof)).to(
        #            device='cuda')

        self.trajectory_np = self.trajectory.cpu().detach().numpy()
        self.velocity_np = self.velocity.cpu().detach().numpy()
        obses = []
        target = []

        for t, pos_vel in enumerate(zip(self.trajectory_np, self.velocity_np)):
            time.sleep(0.1)
            des_pos = self.trajectory_np[t]
            des_vel = self.velocity_np[t]
            #print(des_pos, des_vel)
            #print("controller", self.controller)
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            # ac = np.tanh(ac)
            # print("ac", ac)
            #obses.append(np.array(self.env.sim.data.mocap_pos.copy()).reshape(-1) + 0.05*np.array(ac).reshape(-1))
            obs, rewards, done, info = env.step(ac)
            #obses.append(env.get_body_com("fingertip")[0:2])
            #target.append(env.get_body_com("target")[0:2])
            env.render()
        #print("reward", env.rewards_no_ip)
        target = np.array(env.goal)
        #print("traget", target)
        import matplotlib.pyplot as plt
        plt.plot(target[1:, 0], target[1:, 1])
        plt.xlabel("x axis")
        plt.ylabel("y axis")
        plt.title("2 dimensional trajectory")
        #plt.show()
        '''
        finger = np.array(env.finger)
        target = np.array(env.goal)
        import matplotlib.pyplot as plt
        position_obses = obses[:, -10:-5]
        velocity_obses = obses[:, -5:]
        #position_obses_noise = # obses_noise[:, -10:-5]
        #velocity_obses_noise = # obses_noise[:, -5:]

        for i in range(5):
            plt.plot(position_obses_noise[:, i], label='without noise')
            plt.plot(position_obses[:, i],
                     label='with noise') # label_name(name))
            plt.legend()
            plt.title(f'position_joint_{i}')
            plt.savefig(f'position_joint_{i}')
            plt.cla()

            plt.plot(velocity_obses[:, i], label='without noise')
            plt.plot(velocity_obses_noise[:, i],
                     label='with noise')  # label_name(name))
            plt.legend()
            plt.title(f'velocity_joint_{i}')
            plt.savefig(f'velocity_joint_{i}')
            plt.cla()
        a = 1
        '''