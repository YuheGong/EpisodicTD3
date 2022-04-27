from utils.logger import logging
from utils.yaml import write_yaml, read_yaml
import gym
import numpy as np
import argparse
from stable_baselines3.common.callbacks import EvalCallback
from utils.model import policy_kwargs_building
from model.episodic_td3 import EpisodicTD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise

parser = argparse.ArgumentParser()
parser.add_argument("--e", type=str, help="the environment")
parser.add_argument("--seed", type=str, help="the seed")
args = parser.parse_args()


def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

algo = "promp_td3"
env_id = "ALRReacherBalanceIP-v"

file_name = algo +".yml"
data = read_yaml(file_name)[env_id]
data['env_params']['env_name'] = data['env_params']['env_name'] + args.e

# create log folder
path = logging(data['env_params']['env_name'], data['algorithm'])
data['path'] = path

# make the environment
env = gym.make(data["env_params"]['env_name'])
eval_env = gym.make(data["env_params"]['env_name'])

# make the model and save the model
ALGO = EpisodicTD3
policy_kwargs = policy_kwargs_building(data)
policy = data['algo_params']['policy']
env.reset()

#print("env", env)

model = ALGO(policy, env, seed=1,  initial_promp_params=[2,2], critic_network_kwargs=policy_kwargs, verbose=1, trajectory_noise_sigma=0.3,
                 critic_learning_rate=data["algo_params"]['learning_rate'],
                 policy_delay=2, data_path=data["path"], gamma=0.99)

# csv file path
data["path_in"] = data["path"] + '/' + data['algorithm'].upper() + '_1'
data["path_out"] = data["path"] + '/data.csv'

try:
    eval_env_path = data['path'] + "/eval/"
    #eval_callback = EvalCallback(eval_env, best_model_save_path=eval_env_path,
    #                             n_eval_episodes=data['eval_env']['n_eval_episode'],
    #                             log_path=eval_env_path, eval_freq=data['eval_env']['eval_freq'],
    #                             deterministic=False, render=False)
    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']))
except KeyboardInterrupt:
    write_yaml(data)
    model.save(data["path"] + "/model.zip")
    print('')
    print('training interrupt, save the model and config file to ' + data["path"])
else:
    write_yaml(data)
    model.save(data["path"] + "/model.zip")
    print('')
    print('training FINISH, save the model and config file to ' + data['path'])
