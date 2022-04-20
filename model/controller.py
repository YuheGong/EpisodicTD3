import torch
from typing import Tuple, Union
from gym import Env




class BaseController:
    def __init__(self, env: Env, **kwargs):
        self.env = env

    def get_action(self, des_pos, des_vel):
        raise NotImplementedError

class PosController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        super(PosController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        cur_pos = self.obs()[:3].reshape(-1)
        des_pos = des_pos - cur_pos
        return des_pos, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        cur_pos = observation[:, :3].reshape(-1,self.num_dof)
        des_pos = des_pos - cur_pos
        return des_pos

    def obs(self):
        return self.env.obs_for_promp()


class VelController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        super(VelController, self).__init__(env)

    def get_action(self, des_pos, des_vel):
        return des_vel, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        return des_vel

    def obs(self):
        return self.env.obs_for_promp()

class PDController(BaseController):

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
        self.trq = []
        self.pos = []
        self.vel = []
        super(PDController, self).__init__(env)

    def get_action(self, des_pos, des_vel):#, action_noise=None):
        cur_pos = self.obs()[-2*self.num_dof:-self.num_dof].reshape(self.num_dof)
        cur_vel = self.obs()[-self.num_dof:].reshape(self.num_dof)
        #print("(des_pos - cur_pos", des_pos - cur_pos)
        #print("des_vel - cur_vel", des_vel - cur_vel)
        trq = self.p_g * (des_pos - cur_pos) + self.d_g * (des_vel - cur_vel)
        self.trq.append(trq)
        self.pos.append(cur_pos)
        self.vel.append(cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, observation):
        cur_vel = observation[:, -self.num_dof:].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        return trq

    def obs(self):
        return self.env.obs_for_promp()
