import torch
from typing import Tuple, Union
from gym import Env
import numpy as np

"""
The controllers.

:function get_action:
    Datatype: 
        Numpy
    Input: 
        des_pos: desired reference position
        des_vel: desired reference velocity
    Return:
        the action used for indicating the movements of the robot
        desired reference position
        esired reference velocity

:function predict_action:
    Datatype: 
        Tensor
    Input: 
        des_pos: desired reference position
        des_vel: desired reference velocity
        observation: the observations stored in Replay Buffer
    Return:
        the action used for updating the critic network and ProMP weights
        
:function obs:
    Datatype: 
        Numpy
    Return:
        current observations in the environment
        
"""


class BaseController:
    def __init__(self, env: Env, **kwargs):
        self.env = env
        if "Meta" in str(self.env):
            self.env.obs_for_promp = self.meta_obs

    def get_action(self, des_pos, des_vel, des_acc):
        raise NotImplementedError

    def meta_obs(self):
        return np.concatenate([
            np.array(self.env.sim.data.mocap_quat).reshape(-1),
            np.array(self.env.sim.data.mocap_pos).reshape(-1),
            np.array([1, ]).reshape(-1),
        ]).reshape(-1)

class PosController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        super(PosController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):
        cur_pos = self.obs()[-self.num_dof:].reshape(-1)
        des_pos = des_pos #- cur_pos
        return des_pos, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_pos = observation[:, -self.num_dof:].reshape(-1,self.num_dof)
        des_pos = des_pos #- cur_pos
        return des_pos

    def obs(self):
        return self.env.obs_for_promp()


class VelController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        super(VelController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):
        cur_vel = self.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)
        des_vel = (des_vel - cur_vel) * self.env.dt
        #print("des_vel", des_vel)
        return des_vel, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_vel =  observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        des_vel = des_vel - cur_vel
        return des_vel

    def obs(self):
        return self.env.obs_for_promp()




class PDController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],
                 num_dof: None):
        if isinstance(p_gains, str):
            p_gains = np.fromstring(p_gains, dtype=float, sep=',')
            self.p_gains = torch.Tensor(p_gains).to(device='cuda')
            d_gains = np.fromstring(d_gains, dtype=float, sep=',')
            self.d_gains = torch.Tensor(d_gains).to(device='cuda')
        else:
            self.p_gains = torch.Tensor([p_gains]).to(device='cuda')
            self.d_gains = torch.Tensor([d_gains]).to(device='cuda')
        self.p_g = p_gains
        self.d_g = d_gains
        self.num_dof = num_dof
        self.trq = []
        self.pos = []
        self.vel = []
        super(PDController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):#, action_noise=None):
        cur_pos = self.obs()[-3 * self.num_dof:-2 * self.num_dof].reshape(self.num_dof)
        cur_vel = self.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)

        trq = self.p_g * (des_pos - cur_pos) + self.d_g * (des_vel - cur_vel)
        #trq = self.p_g * (des_pos- cur_pos) + self.d_g * (des_vel )
        self.trq.append(trq)
        self.pos.append(cur_pos)
        self.vel.append(cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_vel = observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -3 * self.num_dof:-2 * self.num_dof].reshape(observation.shape[0], self.num_dof)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        #trq = self.p_gains * (des_pos- cur_pos) + self.d_gains * (des_vel)
        return trq

    def obs(self):
        return self.env.obs_for_promp()


class PIDController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],
                 i_gains: Union[float, Tuple],
                 num_dof: None):
        self.p_gains = torch.Tensor([float(p_gains)]).to(device='cuda')
        self.d_gains = torch.Tensor([float(d_gains)]).to(device='cuda')
        self.i_gains = torch.Tensor([float(i_gains)]).to(device='cuda')
        self.p_g = p_gains
        self.d_g = d_gains
        self.i_g = i_gains
        self.num_dof = num_dof
        self.trq = []
        self.pos = []
        self.vel = []
        super(PIDController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):#, action_noise=None):
        cur_pos = self.obs()[-3*self.num_dof:-2*self.num_dof].reshape(self.num_dof)
        cur_vel = self.obs()[-2*self.num_dof:-self.num_dof].reshape(self.num_dof)
        cur_acc = self.obs()[-self.num_dof:].reshape(self.num_dof)
        #print("(des_pos - cur_pos", des_pos - cur_pos)
        #print("des_vel - cur_vel", des_vel - cur_vel)
        trq = self.p_g * (des_pos - cur_pos) + self.d_g * (des_vel - cur_vel )\
              + self.i_g * (des_acc - cur_acc) / self.env.dt #* self.env.dt
        self.trq.append(trq)
        self.pos.append(cur_pos)
        self.vel.append(cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_acc = observation[:, -self.num_dof:].reshape(observation.shape[0], self.num_dof)
        cur_vel = observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -3 * self.num_dof:-2*self.num_dof].reshape(observation.shape[0], self.num_dof)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * ((des_vel - cur_vel) / self.env.dt + cur_acc)\
              + self.i_gains * (des_acc - cur_acc) / self.env.dt #* self.env.dt
        return trq

    def obs(self):
        return self.env.obs_for_promp()
