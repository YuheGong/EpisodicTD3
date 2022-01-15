import gym
import numpy as np
import numpy as np
import torch as th

from gym import wrappers
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.ppo import MlpPolicy
import os
import pickle
import tensorflow as tf
import alr_envs


def logging(env_name, algorithm):

    init_folder = os.listdir("./logs")

    if algorithm not in init_folder:
        os.makedirs('./logs/' + algorithm)

    env_log_index = env_name.index(':')
    env_log_name = env_name[env_log_index + 1:]
    path = "logs/" + algorithm + '/'
    folders = os.listdir(path)
    folders = [folder for folder in folders if env_log_name in folder]


    if folders == []:
        path = path + env_log_name + "_1"
    else:
        a = 0
        for i in range(999):
            number = [folder[-i - 1:] for folder in folders]
            if not any([n.isdigit() for n in number]):
                folders = [folder for folder in folders if folder[-i:].isdigit() == True]
                a = -i
                break
        s = 0
        for folder in folders:
            if int(folder[a:]) > s:
                s = int(folder[a:])
        s += 1
        path = path + env_log_name + '_' + str(s)
    print('log into: ' + path)
    return path


if __name__ == "__main__":

    def make_env(env_name, rank, seed=0):

        def _init():
            env = gym.make(env_name)
            #env = wrappers.Monitor(env)#, path, force=True)
            return env

        return _init

    env_name = 'alr_envs:ALRBallInACupSimpleDense-v0'
    algorithm = 'ppo'
    path = logging(env_name, algorithm)

    n_cpu = 1
    env = DummyVecEnv(env_fns=[make_env(env_name, i) for i in range(n_cpu)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[algorithm]

    model = ALGO(MlpPolicy, env, verbose=1,
                # policy_kwargs=policy_kwargs,
                tensorboard_log= path,
                learning_rate=0.00001,
                n_steps=2048)
    model.learn(total_timesteps=int(1.2e6))  # , callback=TensorboardCallback())

    # save the model
    model_path = os.path.join(path, "PPO.zip")
    model.save(model_path)
    stats_path = os.path.join(path, "PPO.pkl")
    env.save(stats_path)
