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



class Schedule:

    def __init__(self, env):
        self.env = env

    def schedule(self, env, model):
        raise NotImplementedError

class dmcCheetahDens_v0_schedule(Schedule):

    def schedule(self, model):
        if model.eval_reward > 180:
            model.actor_learning_rate = 1.e-9
            model.actor_optimizer.param_groups[0]['lr'] = model.actor_learning_rate

class dmcHopperDens_v0_schedule(Schedule):

    def schedule(self, model):
        if model.eval_reward > 50:
            model.actor_learning_rate = 0.00001
            model.actor_optimizer.param_groups[0]['lr'] = model.actor_learning_rate
            model.noise_sigma = 0.05
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))
        else:
            model.actor_learning_rate = 0.00005
            model.actor_optimizer.param_groups[0]['lr'] = model.actor_learning_rate
            model.noise_sigma = 0.1
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))

class dmcWalkerDens_v0_schedule(Schedule):

    def schedule(self, model):
       if model.eval_reward > 45:#and model.eval_reward < 50:
            model.actor_learning_rate = 1.e-9
            model.actor_optimizer.param_groups[0]['lr'] = model.actor_learning_rate

class FetchReacher_schedule(Schedule):

    def schedule(self, model):

        if model.eval_reward > -2 and model.eval_reward <= -1:
            model.noise_sigma = 0.1
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))
        elif model.eval_reward > -1:
            model.noise_sigma = 0.01
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))
        else:
            model.noise_sigma = 0.5
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))
            #model.actor_learning_rate = 0.000005
            #model.actor_learning_rate = 0.0001
            #model.actor_optimizer.param_groups[0]['lr'] = model.actor_learning_rate


class MetaPickAndPlace_schedule(Schedule):

    def schedule(self, model):

        if model.eval_reward > 1100:
            model.noise_sigma = 0.1
            model.noise = NormalActionNoise(mean=np.zeros(model.dof), sigma=model.noise_sigma * np.ones(model.dof))

