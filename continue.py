import argparse
from utils.model import model_learn
import os
from utils.env import env_maker, env_save, env_continue_load
from utils.logger import logging
from utils.yaml import write_yaml, read_yaml
import gym
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from model.promp_td3 import ProMPTD3


file_name = "promp_td3.yml"
env_id = "ALRReacherBalance-v1"
model_id = "10"
data = read_yaml(file_name)[env_id]

# create log folder
path = logging(data['env_params']['env_name'], data['algorithm'])
data['path'] = path
data["continue"] = True
data['continue_path'] = "logs/promp_td3/" + env_id + "_" + model_id

# choose the algorithm according to the algo
ALGOS = {
    'a2c': A2C,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo': PPO,
    'td3': TD3,
    'promp_td3': ProMPTD3
}
ALGO = ALGOS[data['algorithm']]


# make the environment
env, eval_env = env_continue_load(data)

# make the model and save the model
model_path = os.path.join(data['continue_path'], 'model')
model = ProMPTD3.continue_load(model_path, tensorboard_log=data['path'], env=env)
#model.set_env(env)

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
    model.save("model")
    print('')
    print('training interrupt, save the model and config file to ' + data["path"])
else:
    write_yaml(data)
    print('')
    model.save("model")
    print('training FINISH, save the model and config file to ' + data['path'])
