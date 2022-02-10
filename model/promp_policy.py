from abc import ABC
from typing import Tuple, Union
import torch
import gym
import numpy as np
from . import det_promp
import torch as th
from mp_env_api.utils.policies import BaseController
from stable_baselines3.common.noise import NormalActionNoise


class PosVelStepController(BaseController):

    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof/2)
        super(PosVelStepController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        cur_pos = self.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)
        des_pos = des_pos #- cur_pos
        return des_pos, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        cur_pos = observation[:, -2 * self.num_dof:-self.num_dof].reshape(-1, self.num_dof)
        des_pos = des_pos #- cur_pos
        return des_pos

    def obs(self):
        return self.env.obs_for_promp()


class PDStepController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],
                 num_dof: None):
        self.p_gains = torch.Tensor([p_gains]).to(device='cuda')
        self.d_gains = torch.Tensor([d_gains]).to(device='cuda')
        self.p_g = p_gains
        self.d_g = d_gains
        self.num_dof = num_dof
        super(PDStepController, self).__init__(env)

    def get_action(self, des_pos, des_vel):#, action_noise=None):
        cur_pos = self.obs()[-2*self.num_dof:-self.num_dof].reshape(self.num_dof)
        cur_vel = self.obs()[-self.num_dof:].reshape(self.num_dof)
        trq = self.p_g * (des_pos - cur_pos) + self.d_g * (des_vel - cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        cur_vel = observation[:, -self.num_dof:].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq

    def obs(self):
        return self.env.obs_for_promp()

class DetPMPWrapper(ABC):
    def __init__(self, env: gym.Wrapper, num_dof: int, num_basis: int, width: int, step_length=None,
                 weights_scale=1, zero_start=False, zero_goal=False, noise_sigma=None,
                 **mp_kwargs):

        self.controller = PDStepController(env, p_gains=mp_kwargs['policy_kwargs']['policy_kwargs']['p_gains'],
                                       d_gains=mp_kwargs['policy_kwargs']['policy_kwargs']['d_gains'], num_dof=num_dof)
        self.controller = PosVelStepController(env, num_dof=num_dof)

        self.weights_scale = torch.Tensor(weights_scale)
        self.trajectory = None
        self.velocity = None

        self.step_length = step_length
        self.env = env
        dt = self.env.dt
        self.noise_sigma = noise_sigma
        self.num_dof = num_dof
        self.mp = det_promp.DeterministicProMP(n_basis=num_basis, n_dof=num_dof, width=width, off=0.01,
                                               zero_start=zero_start, zero_goal=zero_goal, n_zero_bases=0, step_length=self.step_length,
                                               dt=dt)

    def predict_action(self, step, observation):
        self.calculate_traj = self.trajectory[step].reshape(-1, self.num_dof)
        self.calculate_vel = self.velocity[step].reshape(-1, self.num_dof)
        actions = self.controller.predict_actions(self.calculate_traj, self.calculate_vel, observation)
        return actions

    def update(self):
        weights = self.mp.weights
        _,  self.trajectory, self.velocity, __ = self.mp.compute_trajectory(weights)
        #self.trajectory += th.Tensor(self.controller.obs()[-2*self.num_dof:-self.num_dof]).to(device='cuda')
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
        a = action
        weights = a#.cpu().detach().numpy() #+ noise().reshape(self.mp.weights.shape[0], self.mp.weights.shape[1])
        des_pos = np.dot(pos_feature, weights)
        des_vel = np.dot(vel_feature, weights) / self.mp.corrected_scale

        trajectory = des_pos + self.controller.obs()[-2*self.num_dof:-self.num_dof]
        velocity = des_vel
        obses = []
        target = []

        for t, pos_vel in enumerate(zip(trajectory, velocity)):
            time.sleep(0.1)
            des_pos = (trajectory)[t]
            des_vel = (velocity)[t]
            ac, _, __ = self.controller.get_action(des_pos, des_vel)
            # ac = np.tanh(ac)
            # print("ac", ac)
            obs, rewards, done, info = env.step(ac)
            #obses.append(env.get_body_com("fingertip")[0:2])
            #target.append(env.get_body_com("target")[0:2])
            env.render()

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