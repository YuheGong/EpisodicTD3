from abc import ABC
from typing import Tuple, Union
import torch
import gym
import numpy as np
from . import det_promp
import torch as th
from mp_env_api.utils.policies import BaseController
from stable_baselines3.common.noise import NormalActionNoise
class PDStepController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple]):
        self.p_gains = torch.Tensor([p_gains]).to(device='cuda')
        self.d_gains = torch.Tensor([d_gains]).to(device='cuda')
        self.p_g = p_gains
        self.d_g = d_gains
        super(PDStepController, self).__init__(env)

    def get_action(self, des_pos, des_vel):#, action_noise=None):
        cur_pos = self.obs()[-10:-5].reshape(5)
        cur_vel = self.obs()[-5:].reshape(5)
        assert des_pos.shape == cur_pos.shape, \
            f"Mismatch in dimension between desired position {des_pos.shape} and current position {cur_pos.shape}"
        assert des_vel.shape == cur_vel.shape, \
            f"Mismatch in dimension between desired velocity {des_vel.shape} and current velocity {cur_vel.shape}"
        trq = self.p_g * (des_pos - cur_pos) + self.d_g * (des_vel - cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        cur_vel = observation[:, -5:].reshape(observation.shape[0], 5)
        cur_pos = observation[:, -10:-5].reshape(observation.shape[0], 5)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq

    def obs(self):
        return self.env.envs[0].get_obs()


class DetPMPWrapper(ABC):
    def __init__(self, env: gym.Wrapper, num_dof: int, num_basis: int, width: int, step_length=None,
                 weights_scale=1, zero_start=False, zero_goal=False, noise_sigma=None,
                 **mp_kwargs):

        self.controller = PDStepController(env, p_gains=mp_kwargs['policy_kwargs']['policy_kwargs']['p_gains'],
                                       d_gains=mp_kwargs['policy_kwargs']['policy_kwargs']['d_gains'])

        self.weights_scale = torch.Tensor(weights_scale)
        self.trajectory = None
        self.velocity = None

        self.step_length = step_length
        self.env = env
        dt = self.env.envs[0].dt
        self.noise_sigma = noise_sigma
        self.num_dof = num_dof
        self.mp = det_promp.DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width, off=0.01,
                                               zero_start=zero_start, zero_goal=zero_goal, n_zero_bases=2, step_length=self.step_length,
                                               dt=dt)

    def predict_action(self, step, observation):
        self.calculate_traj = self.trajectory[step].reshape(-1,self.num_dof)
        self.calculate_vel = self.velocity[step].reshape(-1,self.num_dof)
        actions = self.controller.predict_actions(self.calculate_traj, self.calculate_vel, observation)
        return actions

    def update(self):
        weights = self.mp.weights
        _,  self.trajectory, self.velocity, __ = self.mp.compute_trajectory(weights)
        self.trajectory += th.Tensor(self.obs()[-10:-5]).to(device='cuda')
        self.trajectory_np = self.trajectory.cpu().detach().numpy()
        self.velocity_np = self.velocity.cpu().detach().numpy()


    def get_action(self, timesteps):
        """ This function generates a trajectory based on a DMP and then does the usual loop over reset and step"""
        n_actions =(self.num_dof,)  # env.action_space.shape[-1]
        noise_dist = NormalActionNoise(mean=np.zeros(n_actions), sigma=self.noise_sigma * np.ones(n_actions))

        trajectory = self.trajectory_np[timesteps] + noise_dist()
        velocity = self.velocity_np[timesteps]

        action, des_pos, des_vel = self.controller.get_action(trajectory, velocity)
        return action

    def obs(self):
        return self.env.envs[0].get_obs()

    def eval_rollout(self, env, a):
        rewards = 0
        #self.plot_pos = np.zeros((200,5))
        #self.plot_vel = np.zeros((200, 5))

        for t, pos_vel in enumerate(zip(self.trajectory_np, self.velocity_np)):
            des_pos = pos_vel[0]
            des_vel = pos_vel[1]
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            ac = np.clip(ac, -1, 1).reshape(1,5)
            obs, reward, done, info = env.step(ac)
            #self.plot_pos[t,:] = obs[:, -10:-5].reshape(-1)
            #self.plot_vel[t,:] = obs[:, -5:].reshape(-1)
            rewards += reward[0]
        return env.envs[0].rewards_no_ip


    def load(self, action):
        action = torch.FloatTensor(action)
        params = action.reshape(self.mp.n_basis, self.mp.n_dof) * self.weights_scale
        self.mp.weights = params.to(device="cuda")
        _, des_pos, des_vel, __ = self.mp.compute_trajectory(self.mp.weights)
        #des_pos += th.Tensor(self.obs()[-10:-5]).to(device='cuda')
        return des_pos, des_vel


    def render_rollout(self, action, env, noise):
        import time

        self.trajectory, self.velocity = self.load(action)
        #self.update_tra_with_noise(noise())

        a = action
        weights = a#.cpu().detach().numpy() #+ noise().reshape(self.mp.weights.shape[0], self.mp.weights.shape[1])
        _, des_pos, des_vel, __ = self.mp.compute_trajectory_with_noise(weights)
        self.trajectory = des_pos #+ noise()
        self.velocity = des_vel
        self.trajectory += self.obs()[-10:-5]

        #trajectory = self.trajectory.cpu().detach().numpy()
        #velocity = self.velocity.cpu().detach().numpy()

        # if timesteps == 0:
        #n_actions = (50,5)
        #noise_dist = NormalActionNoise(mean=np.zeros(n_actions),
        #                               sigma=0.1 * np.ones(n_actions))
        #noise = noise_dist()
        #_, noise_traj, noise_vel, __ = self.mp.compute_trajectory_with_noise(noise)

        #self.trajectory_noise = self.trajectory +noise_traj
        #self.velocity_noise = self.velocity+ noise_vel
        #trajectory = self.trajectory_noise
        #velocity = self.velocity_noise
        #env.reset()
        obses = []
        '''
        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            time.sleep(0.1)
            #print("t", t)
            print("original", t, pos_vel[0])
            des_pos = pos_vel[0]
            print("addnoise", des_pos)
            des_vel = pos_vel[1] #+ noise()
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            #print("ac_origin", ac)
            ac = np.clip(ac, -1, 1) #+ noise()
            #ac = np.tanh(ac)
            print("ac", ac)
            obs, rewards, done, info = env.step(ac)
            obses.append(obs)
            #env.render()
        '''
        self.trajectory_noise = self.trajectory  # + noise_traj
        self.velocity_noise = self.velocity  # + noise_vel
        trajectory = self.trajectory_noise
        velocity = self.velocity_noise
        env.reset()

        obses_noise = []

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            time.sleep(0.1)
            #print("t", t)

            print("original", t, pos_vel[0])
            #n_actions = (50, 5)
            #noise_dist = NormalActionNoise(mean=np.zeros(n_actions),
            #                               sigma=0.3 * np.ones(n_actions))
            #noise = noise_dist()
            # _, noise_traj, noise_vel, __ = self.mp.compute_trajectory_with_noise(noise)
            trajectory = trajectory #+ noise_traj
            velocity = velocity# + noise_traj
            des_pos = (trajectory)[t]
            print("addnoise", des_pos)
            des_vel = (velocity)[t]
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            #print("ac_origin", ac)
            ac = np.clip(ac, -1, 1) #+ noise()
            #ac = np.tanh(ac)
            print("ac", ac)
            obs_noise, rewards, done, info = env.step(ac)
            obses_noise.append(obs_noise)
            env.render()
        obses = np.array(obses)
        obses_noise = np.array(obses_noise)
        import matplotlib.pyplot as plt
        position_obses = obses[:, -10:-5]
        velocity_obses = obses[:, -5:]
        position_obses_noise = obses_noise[:, -10:-5]
        velocity_obses_noise = obses_noise[:, -5:]

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