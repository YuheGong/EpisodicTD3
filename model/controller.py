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
    def __init__(self, env: Env, p_gains=None, d_gains=None, **kwargs):
        self.env = env
        if p_gains is not None:
            if isinstance(p_gains, str):
                p_gains = np.fromstring(p_gains, dtype=float, sep=',')
                self.p_gains = torch.Tensor(p_gains).to(device='cuda')
            else:
                self.p_gains = p_gains * torch.ones(self.num_dof).to(device='cuda')
        if d_gains is not None:
            if isinstance(d_gains, str):
                d_gains = np.fromstring(d_gains, dtype=float, sep=',')
                self.d_gains = torch.Tensor(d_gains).to(device='cuda')
            else:
                self.d_gains = d_gains * torch.ones(self.num_dof).to(device='cuda')


    def get_action(self, des_pos, des_vel, des_acc):
        raise NotImplementedError


class PosController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        super(PosController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):
        #cur_pos =self.env.current_pos().reshape(-1)
        des_pos = des_pos #- cur_pos
        return des_pos, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        #cur_pos = observation[:, -self.num_dof:].reshape(-1,self.num_dof)
        des_pos = des_pos # - cur_pos
        return des_pos


class MetaWorldController(BaseController):
    """
    A Metaworld Controller. Using position and velocity information from a provided environment,
    the controller calculates a response based on the desired position and velocity.
    Unlike the other Controllers, this is a special controller for MetaWorld environments.
    They use a position delta for the xyz coordinates and a raw position for the gripper opening.
    :param env: A position environment
    """

    def __init__(self,
                 env: None,
                 num_dof: None):
        self.num_dof = int(num_dof)
        self.env = env
        super(MetaWorldController, self).__init__(env)

    def get_action(self, des_pos, des_vel, des_acc):
        gripper_pos = des_pos[-1]
        g_pos = self.env.current_pos()[-1]
        cur_pos = self.env.current_pos()[-self.num_dof:-1]
        xyz_pos = des_pos[:-1] #* self.env.action_scale
        #trq = np.hstack([(xyz_pos -cur_pos), gripper_pos])
        trq = np.hstack([(xyz_pos-cur_pos), gripper_pos-g_pos])
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        gripper_pos = des_pos[:, -1]
        g_pos = observation[:, -1].reshape(-1,1)
        cur_pos = observation[:, -self.num_dof:-1].reshape(-1,self.num_dof-1)
        xyz_pos = des_pos[:, :-1] #* self.env.action_scale
        #trq = torch.hstack([(xyz_pos -cur_pos), gripper_pos.reshape(-1,1)])
        trq = torch.hstack([(xyz_pos-cur_pos), (gripper_pos.reshape(-1,1)-g_pos)])
        return trq

class VelController(BaseController):
    def __init__(self,
                 env: None,
                 num_dof: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],):
        self.num_dof = int(num_dof)
        super(VelController, self).__init__(env, p_gains=p_gains, d_gains=d_gains)

    def get_action(self, des_pos, des_vel, des_acc):
        cur_vel = self.obs()[-2 * self.num_dof:-self.num_dof].reshape(self.num_dof)
        cur_pos = self.obs()[-3 * self.num_dof:-2 * self.num_dof].reshape(self.num_dof)
        #des_vel = (des_vel - cur_vel) * self.d_gains.cpu().detach().numpy()
        #des_vel = (des_vel ) * self.d_gains.cpu().detach().numpy()
        #print("des_vel", des_vel)
        self.env.env.physics.data.qfrc_bias[3:].copy().reshape(-1),
        self.env.env.physics.data.qM.copy().reshape(-1),
        self.env.env.physics.data.efc_force.copy().reshape(-1),
        self.env.env.physics.data.efc_J.copy().reshape(-1),


        #c = self.obs()[:self.num_dof].reshape(self.num_dof)
        #M = self.obs()[self.num_dof:self.num_dof+self.num_dof*self.num_dof].reshape(self.num_dof,self.num_dof)
        v_point = (des_pos-cur_pos) #/ self.env.env.dt #- cur_vel #/ self.env.env.dt
        #f = self.obs()[self.num_dof+self.num_dof*self.num_dof:self.num_dof+self.num_dof*self.num_dof+500]
        #J = self.obs()[self.num_dof+self.num_dof*self.num_dof+500:self.num_dof+self.num_dof*self.num_dof+5000]
        des_vel = v_point#(M.reshape(6,6) @ v_point.reshape(6,1)).reshape(-1) + c #- (J.reshape(9,500) @ f.reshape(500,1))[3:].reshape(-1)
        return des_vel, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_vel = observation[:, -2 * self.num_dof:-self.num_dof].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -3 * self.num_dof:-2 * self.num_dof].reshape(observation.shape[0], self.num_dof)
        #des_vel = (des_vel) * self.d_gains

        #c = observation[:,:self.num_dof].reshape(-1,self.num_dof)
        #M = observation[:,self.num_dof:self.num_dof + self.num_dof * self.num_dof].reshape(-1, self.num_dof,self.num_dof)
        v_point = (des_pos-cur_pos) #/ self.env.env.dt- cur_vel ) #/ self.env.env.dt
        #f = observation[:,
        #    self.num_dof + self.num_dof * self.num_dof:self.num_dof + self.num_dof * self.num_dof + 500]
        #J = observation[:,
        #    self.num_dof + self.num_dof * self.num_dof + 500:self.num_dof + self.num_dof * self.num_dof + 5000]
        des_vel = v_point#(M @ v_point.reshape(-1,6,1)).reshape(-1, 6) + c#\
                  #- (J.reshape(-1,9, 500) @ f.reshape(-1,500, 1))[:, 3:, :].reshape(-1, 6)
        return des_vel


class PDController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],
                 num_dof: None):
        self.num_dof = num_dof
        self.trq = []
        self.pos = []
        self.vel = []
        super(PDController, self).__init__(env, p_gains, d_gains)

    def get_action(self, des_pos, des_vel, des_acc):#, action_noise=None):
        cur_pos = self.env.current_pos().reshape(self.num_dof)
        cur_vel = self.env.current_vel().reshape(self.num_dof)

        trq = self.p_gains.cpu().detach().numpy() * (des_pos - cur_pos) \
              + self.d_gains.cpu().detach().numpy() * (des_vel - cur_vel)
        self.trq.append(trq)
        self.pos.append(cur_pos)
        self.vel.append(cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        cur_vel = observation[:, -1 * self.num_dof:].reshape(observation.shape[0], self.num_dof)
        cur_pos = observation[:, -2 * self.num_dof:-1 * self.num_dof].reshape(observation.shape[0], self.num_dof)
        #cur_vel = observation[:, -2 * self.num_dof:-1 * self.num_dof].reshape(observation.shape[0], self.num_dof)
        #cur_pos = observation[:, -3 * self.num_dof:-2 * self.num_dof].reshape(observation.shape[0], self.num_dof)
        trq = self.p_gains * (des_pos - cur_pos) + self.d_gains * (des_vel - cur_vel)
        #trq = self.p_gains * (des_pos- cur_pos) + self.d_gains * (des_vel)
        return trq

    def obs(self):
        return self.env.current_pos()


class PIDController(BaseController):

    def __init__(self,
                 env: None,
                 p_gains: Union[float, Tuple],
                 d_gains: Union[float, Tuple],
                 i_gains: Union[float, Tuple],
                 num_dof: None):
        self.num_dof = num_dof
        if isinstance(i_gains, str):
            i_gains = np.fromstring(i_gains, dtype=float, sep=',')
            self.i_gains = torch.Tensor(i_gains).to(device='cuda')
        else:
            self.i_gains = i_gains * torch.ones(self.num_dof).to(device='cuda')
        if self.i_gains.requires_grad == False:
            self.i_gains.requires_grad = True

        self.trq = []
        self.pos = []
        self.vel = []
        super(PIDController, self).__init__(env, p_gains, d_gains)

    def get_action(self, des_pos, des_vel, des_acc):#, action_noise=None):

        trq = self.p_gains.cpu().detach().numpy() * (des_pos) + self.d_gains.cpu().detach().numpy() * (des_vel)
        #self.pos.append(cur_pos)
        #self.vel.append(cur_vel)
        return trq, des_pos, des_vel

    def predict_actions(self, des_pos, des_vel, des_acc, observation):
        trq = self.p_gains * (des_pos) +  self.d_gains * (des_vel)
        return trq

