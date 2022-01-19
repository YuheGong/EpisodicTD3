import os
import time
from utils.yaml import read_yaml
import gym
import numpy as np
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from model.promp_td3 import ProMPTD3

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

algo = "promp_td3"
env_id = "ALRReacherBalance-v1"
file_name = algo +".yml"
data = read_yaml(file_name)[env_id]

# create log folder
path = "logs/promp_td3/ALRReacherBalance-v1_28"
data['path'] = path

# make the environment
stats_file = 'env_normalize.pkl'
stats_path = os.path.join(path, stats_file)
env = gym.make(data["env_params"]['env_name'])
algo_path = path + "/algo_mean.npy"
#algo_path = path + "/best_model.npy"

algorithm = np.load(algo_path)

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
policy = data['algo_params']['policy']

model = ALGO(policy, env, verbose=1, create_eval_env=True,
                 tensorboard_log=data['path'],
                 learning_rate=data["algo_params"]['learning_rate'],
                 batch_size=data["algo_params"]['batch_size'], train_freq=200, gradient_steps=200,
                 policy_delay=2, data_path=data["path"])

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
    model.load(algorithm, env, noise)