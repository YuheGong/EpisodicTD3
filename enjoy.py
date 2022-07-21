import os
import time
from utils.yaml import read_yaml
import gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from model.episodic_td3 import EpisodicTD3
from utils.env import env_maker, env_save, env_continue_load

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

algo = "episodic_td3"

env = "FetchReacher-v0"
env = "ALRReacherBalanceIP-v3"
#env = "dmcCheetahDense-v0"
env = 'Meta-basketball-v2'
env = 'Meta-door-open-v2'
env = 'HopperXYJumpMiddle-v0'
#env = "MetaBottonPress-v0"
env_id = env

path = "logs/episodic_td3/" + env + "_1"

file_name = algo +".yml"
if 'Meta' in env_id:
    data = read_yaml(file_name)['Meta-v2']
    data['env_params']['env_name'] =  data['env_params']['env_name'] + ":" + env
else:
    data = read_yaml(file_name)[env_id]

# create log folder
data['path'] = path
#data['continue_path'] = "logs/promp_td3/" + env + "_13"

# make the environment
env = gym.make(data["env_params"]['env_name'])
algo_path = path + "/best_model.npz"
#algo_path = path + "/algo_mean.npz"

algorithm = np.load(algo_path, encoding='bytes', allow_pickle=True)
for i in algorithm:
    algorithm = np.array(algorithm[i])


pos = path + "/pos_features.npz"
vel = path + "/vel_features.npz"

pos_feature= np.load(pos, encoding='bytes', allow_pickle=True)
for i in pos_feature:
    pos_feature = np.array(pos_feature[i])
vel_feature= np.load(vel, encoding='bytes', allow_pickle=True)
for i in vel_feature:
    vel_feature = np.array(vel_feature[i])
print("pos_feature",pos_feature)
# make the model and save the model
ALGO = EpisodicTD3

critic = data['algo_params']['policy']
promp_policy_kwargs = data['promp_params']
print(env)
#model_path = os.path.join(path, 'model')
#model = EpisodicTD3.continue_load(model_path,  env=env)
model = EpisodicTD3(critic, env,
             initial_promp_params=data["algo_params"]['initial_promp_params'],
             verbose=1,
             noise_sigma=data["algo_params"]['noise_sigma'],
             promp_policy_kwargs=promp_policy_kwargs,
             critic_learning_rate=data["algo_params"]['critic_learning_rate'],
             actor_learning_rate=data["algo_params"]['actor_learning_rate'],
             basis_num=data['promp_params']['num_basis'],
             data_path=data["path"])

n_actions = env.action_space.shape[-1]
noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0 * np.ones(n_actions))

#print("algo", algorithm)
basis_num = 10
#algorithm = -1 * np.ones((basis_num, env.action_space.shape[0]))
#algorithm[:, 0] = 10 * np.ones(algorithm[:, 1].shape)
#algorithm[:, 2] = -0.01 #* np.ones(algorithm[:, 2].shape)
#algorithm[10:, 2] = 1 #* np.ones(algorithm[:, 2].shape)
#algorithm[30:, 2] = -1
#algorithm[50:, 2] = 1
#algorithm[70:, 2] = -1
#algorithm[70:, 2] = -1
#algorithm[110:, 2] = -1
#algorithm[90:, 2] = 1
#algorithm[150:, 2] = -1
#algorithm[170:, 2] = -1
#algorithm[190:, 2] = -1
basis_num = data['promp_params']['num_basis']

print("algorithm", algorithm)

model.load(algorithm, env, pos=pos_feature, vel=vel_feature)