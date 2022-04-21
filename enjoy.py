import os
import time
from utils.yaml import read_yaml
import gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from model.promp_td3 import ProMPTD3
from utils.env import env_maker, env_save, env_continue_load

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

algo = "promp_td3"

#env_id = "FetchReacher-v"
#env_id = "ALRReacherBalanceIP-v"
#env = env_id + '3'
env = "Ant-v0"
env_id = env
path = "logs/promp_td3/" + env + "_23"

#env_id = "ALRReacherBalance-v3"
#path = "logs/promp_td3/ALRReacherBalance-v3_2"

#env_id = "ALRReacherBalance-v1"
#path = "logs/promp_td3/ALRReacherBalance-v1_1"

file_name = algo +".yml"
data = read_yaml(file_name)[env_id]

# create log folder
data['path'] = path
#data['continue_path'] = "logs/promp_td3/" + env + "_13"

# make the environment
stats_file = 'env_normalize.pkl'
stats_path = os.path.join(path, stats_file)
env = gym.make("alr_envs:" + env)
algo_path = path + "/best_model.npz"
a = path + "/pos_features.npz"
print("path",algo_path)
algorithm = np.load("logs/promp_td3/Ant-v0_23/algo_mean.npz")#"logs/promp_td3/Ant-v0_1/best_model.npy", allow_pickle=True)
pos_feature = np.load(a)
vel_feature = np.load(path + "/vel_features.npz")
algo_path = path + "/algo_mean.npz"

algorithm = np.load(algo_path)
for i in algorithm:
    algorithm = np.array(algorithm[i])

for i in pos_feature:
    pos_feature = np.array(pos_feature[i])

for i in vel_feature:
    vel_feature = np.array(vel_feature[i])
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
        'promp_td3': ProMPTD3
}
ALGO = ALGOS[algo]
critic = data['algo_params']['policy']
promp_policy_kwargs = data['promp_params']
print(env)
#assert 1==123
#nv, eval_env = env_continue_load(data)
#model_path = os.path.join(data['continue_path'], 'model')
#model = ProMPTD3.continue_load(model_path, tensorboard_log=data['path'], env=env)

model = ALGO(critic, env, seed=1,  initial_promp_params=0.1,  verbose=1,
             trajectory_noise_sigma=0.3, promp_policy_kwargs=promp_policy_kwargs,
             critic_learning_rate=data["algo_params"]['critic_learning_rate'],
             actor_learning_rate=data["algo_params"]['actor_learning_rate'],
             policy_delay=2, data_path=data["path"], gamma=0.99)

if data['algorithm'] == "td3":
    print("td3")
    for i in range(int(200)):
        time.sleep(0.01)
        action, _states = model.load(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
    env.close()
elif data['algorithm'] == "promp_td3":
    #algorithm = np.zeros(25).reshape(5,5)
    n_actions = env.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0 * np.ones(n_actions))
    #noise = action_noise()
    #algorithm += noise.reshape(7, 5)
    #print("noise", noise)
    print("algo", algorithm)
    #assert 1==123
    model.load(algorithm, env, noise, pos_feature=pos_feature, vel_feature=vel_feature)