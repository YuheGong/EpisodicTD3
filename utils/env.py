import gym
import os
from gym import wrappers
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

def env_maker(data: dict, num_envs: int, training=True, norm_reward=True):
    if data["env_params"]['wrapper'] == "VecNormalize":
        env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'][11:-2], data['path'], i) for i in range(num_envs)])
        env = VecNormalize(env, training = training, norm_obs=True, norm_reward=norm_reward)
    else:
        env = gym.make(data["env_params"]['env_name'])
    return env

def model_save(data: dict, model, env, eval_env):
    model_path = os.path.join(data['path'],  "model.zip")
    model.save(model_path)

def env_save(data: dict, model, env, eval_env):
    if 'VecNormalize' in data['env_params']['wrapper']:
        # save env
        stats_path = os.path.join(data['path'], "env_normalize.pkl")
        env.save(stats_path)
        # save evaluation env
        eval_stats_path = os.path.join(data['path'], "eval_env_normalize.pkl")
        eval_env.save(eval_stats_path)

def env_continue_load(data: dict):
    env = gym.make(data["env_params"]['env_name'])
    eval_env = None
    return env, eval_env