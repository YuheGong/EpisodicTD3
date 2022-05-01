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

#env_id = "FetchReacher-v"
#env_id = "ALRReacherBalanceIP-v"
#env = env_id + '3'
#env = "InvertedDoublePendulum-v0"
env = "Ant-v1"
env = "dmcWalkerDense-v0"
#env = "dmcHopperDense-v0"
#env = "dmcCheetahDense-v0"
#env = "MetaButtonPress-v2"
#env = "ALRHalfCheetahJump-v0"
#nv = "Hopper-v0"

#env = "Ant-v0"
env_id = env

path = "logs/episodic_td3/" + env + "_2"

file_name = algo +".yml"
data = read_yaml(file_name)[env_id]
data['env_params']['env_name'] = data['env_params']['env_name']

# create log folder
data['path'] = path
#data['continue_path'] = "logs/promp_td3/" + env + "_13"

# make the environment
env = gym.make(data["env_params"]['env_name'])
algo_path = path + "/best_model.npz"
a = path + "/pos_features.npz"
algo_path = path + "/best_model.npy.npz"
algo_path = path + "/algo_mean.npz"
#a = []
import pickle
#with open("my_file.pkl", "wb") as h:
#    pickle.dump(a, h)
#open(algo_path, 'rb').close()

#print("a",h)
#data = np.array(algo_path )
algorithm = np.load(algo_path, encoding='bytes', allow_pickle=True)
for i in algorithm:
    algorithm = np.array(algorithm[i])

#for i in pos_feature:
#    pos_feature = np.array(pos_feature[i])

#for i in vel_feature:
#    vel_feature = np.array(vel_feature[i])
#"alr_envs:" + env

# make the model and save the model
ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3,
        'episodic_td3': EpisodicTD3
}
ALGO = ALGOS[algo]

critic = data['algo_params']['policy']
promp_policy_kwargs = data['promp_params']
print(env)

model = ALGO(critic, env, seed=1,  initial_promp_params=0.1,  verbose=1,
             noise_sigma=0, promp_policy_kwargs=promp_policy_kwargs,
             critic_learning_rate=data["algo_params"]['critic_learning_rate'],
             actor_learning_rate=data["algo_params"]['actor_learning_rate'], basis_num=data['promp_params']['num_basis'],
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
#
"""
algorithm = 1 * np.ones((basis_num, env.action_space.shape[0]))

for i in range(algorithm.shape[0]):
    if i % 2 == 0:
        algorithm[i, 1] = -1 * np.ones(algorithm[i,1].shape)
        algorithm[i, 2] = -0.1 * np.ones(algorithm[i, 1].shape)
        algorithm[i, 3] = -0.001 * np.ones(algorithm[i, 1].shape)
        algorithm[i, 4] = -1 * np.ones(algorithm[i, 1].shape)
        algorithm[i, 5] = -1 * np.ones(algorithm[i, 1].shape)
"""
#for i in range(algorithm.shape[1]):
#    if i % 2 ==0:
#        algorithm[:, i] = 10 * np.ones(algorithm[:, 1].shape)
#    else:
#        algorithm[:, i] = -10 * np.ones(algorithm[:, 1].shape)

#algorithm[:, :3] = -1 #* np.ones(algorithm[:, :].shape)
#algorithm[:, 3:] = -1

#algorithm = np.random.rand(basis_num, env.action_space.shape[0])
#algorithm = 10 * np.ones(algorithm.shape)
#algorithm[3, :] = -10 * np.ones(algorithm[1, :].shape)
#algorithm[5, :] = -10 * np.ones(algorithm[1, :].shape)
#algorithm[:, 6] = 1.22 * np.ones(algorithm[:, 2].shape)
#algorithm[:, 7] = 0.53 * np.ones(algorithm[:, 2].shape)
#algorithm = np.random.rand(basis_num, env.action_space.shape[0])
#algorithm = 0 * np.ones(shape=algorithm.shape)


#algorithm[:, 2:3] = 0.1 * np.ones(algorithm[:, 2:3].shape)
#algorithm[:, :2] = 0.1# * np.ones(algorithm[:, :2].shape)
#algorithm[2:, :2] = 0.1 * np.ones(algorithm[2:, :2].shape)
#algorithm[4:, :2] = 0.1* np.ones(algorithm[4:, :2].shape)
#algorithm = 1 * np.ones(algorithm.shape)
#algorithm[:, :2] = -0.3 * np.ones(algorithm[:, :2].shape)
#algorithm = 1 * np.ones(algorithm.shape)
#algorithm[:,:2] = 0.1 * np.ones(algorithm[:,:2].shape)
#algorithm[:, :2] = -0.3 * np.ones(algorithm[:, :2].shape)#

print("algorithm", algorithm)
model.load(algorithm, env)