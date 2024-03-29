from utils.logger import logging
from utils.yaml import write_yaml, read_yaml
import gym
import argparse
from utils.model import policy_kwargs_building
from model.episodic_td3 import EpisodicTD3
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from model.schedule import dmcCheetahDens_v0_schedule, \
    dmcHopperDens_v0_schedule, dmcWalkerDens_v0_schedule,FetchReacher_schedule,\
    MetaPickAndPlace_schedule
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env.obs_dict_wrapper import  ObsDictWrapper

def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

''''
env_id = "dmcWalkerDense-v0"
#env_id = "dmcHopperDense-v0"
#env_id = "dmcSwimmerDense-v0"
#env_id = "dmcCheetahDense-v0"
#env_id = "dmcSwimmerDense-v0"
#env_id = "InvertedDoublePendulum-v0"
#env_id = "MetaButtonPress-v2"
#env_id = "Ant-v1"
#env_id = "FetchReacher-v1"
#env_id = "ALRReacherBalanceIP-v3"
#env_id = "ALRHalfCheetahJump-v0"
#env_id = "Hopper-v0"
'''

parser = argparse.ArgumentParser()
parser.add_argument("--env", type=str, help="the environment")
parser.add_argument("--seed", type=str, help="the seed")
parser.add_argument("--con", type=str, help="context")
args = parser.parse_args()


# load yaml file
algo = "episodic_td3"
if int(args.con) == 1:
    file_name = "context.yml"
else:
    file_name = "non_context.yml"

if "Meta" in args.env:
    data = read_yaml(file_name)["Meta-v2"]
else:
    data = read_yaml(file_name)[args.env]

if "Meta" in args.env:
    data['env_params']['env_name'] = data['env_params']['env_name'] + ":" + args.env


# create log folder
path = logging(data['env_params']['env_name'], data['algorithm'])
data['path'] = path
promp_policy_kwargs = data['promp_params']


# make the environment
if "Hopper" in data["env_params"]['env_name']:
    env = gym.make(data["env_params"]['env_name'])
    eval_env = gym.make(data["env_params"]['env_name'])
elif "Meta" in data["env_params"]['env_name']:
    env = gym.make(data["env_params"]['env_name'], seed=int(args.seed))
    eval_env = gym.make(data["env_params"]['env_name'], seed=int(args.seed))
else:
    env = gym.make(data["env_params"]['env_name'])
    eval_env = gym.make(data["env_params"]['env_name'])

# learning rate and noise schedule
Schedule = {
        'FetchReacher-v0': FetchReacher_schedule,
        'MetaPickAndPlace-v0': MetaPickAndPlace_schedule,
}

if args.env in Schedule.keys():
    schedule = Schedule[args.env](env=env)
else:
    schedule = None


def make_env(env_name, path, rank, seed=0):
    def _init():
        env = gym.make(env_name)
        return env
    return _init

#env = DummyVecEnv(env_fns=[make_env(data["env_params"]['env_name'], data['path'], i) for i in range(1)])
#env = VecNormalize(env, training = True, norm_obs=True, norm_reward=True)

# load critic network type and architecture
critic_kwargs = policy_kwargs_building(data)
critic = data['algo_params']['policy']
data['seed'] = args.seed
#data["algo_params"]['initial_promp_params'] = np.ones(30).reshape(5,6)
#data["algo_params"]['initial_promp_params'][:,:3] = -1

# build the model
env.reset()
model = EpisodicTD3(critic, env,
             initial_promp_params=data["algo_params"]['initial_promp_params'],
             schedule=schedule,
             seed=int(args.seed),
             critic_network_kwargs=critic_kwargs,
             verbose=1,
             noise_sigma=data["algo_params"]['action_noise_sigma'],
             promp_policy_kwargs=promp_policy_kwargs,
             critic_learning_rate=data["algo_params"]['critic_learning_rate'],
             actor_learning_rate=data["algo_params"]['actor_learning_rate'],
             basis_num=data['promp_params']['num_basis'],
             data_path=data["path"],
             contextual=args.con,
             weight_noise_judge=data["algo_params"]["weight_noise_judge"],
             weight_noise=data["algo_params"]["weight_noise"],
)


# csv file path
data["path_in"] = data["path"] + '/' + data['algorithm'].upper() + '_1'
data["path_out"] = data["path"] + '/data.csv'


# train and save the model
try:
    eval_env_path = data['path'] + "/eval/"
    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']))
except KeyboardInterrupt:
    write_yaml(data)
    #model.save(data["path"] + "/model.zip")
    print('')
    print('training interrupt, save the model and config file to ' + data["path"])
else:
    write_yaml(data)
    #model.save(data["path"] + "/model.zip")
    print('')
    print('training FINISH, save the model and config file to ' + data['path'])
