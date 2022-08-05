import numpy as np
from utils.callback import ALRBallInACupCallback,DMbicCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import PPO, A2C, DQN, HER, SAC, TD3, DDPG
import torch as th
from .callback import callback_building

def model_building(data, env, seed=None):
    ALGOS = {
        'a2c': A2C,
        'dqn': DQN,
        'ddpg': DDPG,
        'her': HER,
        'sac': SAC,
        'ppo': PPO,
        'td3': TD3
    }
    ALGO = ALGOS[data['algorithm']]

    if "policy_kwargs" in data["algo_params"]:
        policy_kwargs = policy_kwargs_building(data)
    else:
        policy_kwargs = None

    if "special_policy" in data['algo_params']:
        policy = POLICY[data['algo_params']['special_policy']]
    else:
        policy = data['algo_params']['policy']

    if data['algorithm'] == "ppo":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'],
                     n_steps=data["algo_params"]['n_steps'])
    elif data['algorithm'] == "sac":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     train_freq=data["algo_params"]["train_freq"],
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'])
    elif data['algorithm'] == "ddpg":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, create_eval_env=True,
                     tensorboard_log=data['path'],
                     seed=seed,
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'])
    elif data['algorithm'] == "td3":
        model = ALGO(policy, env, policy_kwargs=policy_kwargs, verbose=1, create_eval_env=True,
                     tensorboard_log=data['path'],
                     learning_rate=data["algo_params"]['learning_rate'],
                     batch_size=data["algo_params"]['batch_size'])
    else:
        print("the model initialization function for " + data['algorithm'] + " is still not implemented.")

    return model


def model_learn(data, model, test_env, test_env_path):
    # choose the tensorboard callback function according to the environment wrapper
    CALLBACKS = {
            'ALRBallInACupCallback': ALRBallInACupCallback(),
            'DMbicCallback': DMbicCallback()
        }
    if 'special_callback' in data['algo_params']:
        callback = CALLBACKS[data['algo_params']['special_callback']]
    else:
        callback = None
    eval_callback = callback_building(env=test_env, path=test_env_path, data=data)

    model.learn(total_timesteps=int(data['algo_params']['total_timesteps']), callback=eval_callback)
                #, eval_freq=500, n_eval_episodes=10, eval_log_path=test_env_path, eval_env=test_env)

def cmaes_model_training(algorithm, env, success_full, success_mean, path, log_writer, opts, t, env_id = None):
    fitness = []
    print("----------iter {} -----------".format(t))
    solutions = np.vstack(algorithm.ask())
    #print("solutions_shape", solutions.shape)
    #print("solutions", solutions)
    #print("env",env)
    import torch
    #torch.nn.init.xavier_uniform(env.dynamical_net.weight)

    for i in range(len(solutions)):
        env.reset()
        _, reward, done, infos = env.step(solutions[i])
        if "DeepMind" in env_id:
            success_full.append(env.env.success)
        print('reward', -reward)
        fitness.append(-reward)

        #env.optimizer.zero_grad()

    env.reset()

    '''
    import torch
    print("infos",infos["trajectory"].shape)
    print("actions", infos['step_actions'].shape)
    print("observations", infos['step_observations'].shape)
    loss = np.sum(infos["trajectory"] - infos['step_observations'],axis=1)
    print("shape", loss.shape)
    
    loss = torch.mean(torch.Tensor(loss))
    import tensorflow as tf
    #loss = tf.Variable(loss, requires_grad=True)
    loss_func = torch.nn.MSELoss()
    from torch.autograd import Variable
    #x = torch.unsqueeze(
    x = torch.unsqueeze(torch.Tensor(infos["trajectory"]), dim=1)
    y = torch.unsqueeze(torch.Tensor(infos['step_observations']), dim=1)
    x.requires_grad_()
    y.requires_grad_()
    from torch.autograd import Variable
    #x, y = (x, y)
    loss = loss_func(x, y)
    #loss.requres_grad = True
    #loss_func = torch.nn.MSELoss()
    #loss = loss_func(loss)
    loss.backward()
    env.optimizer.step()
    '''


    algorithm.tell(solutions, fitness)
    _, opt, __, ___ = env.step(algorithm.mean)

    np.save(path + "/algo_mean.npy", algorithm.mean)
    print("opt", -opt)
    opts.append(opt)
    t += 1
    if "DeepMind" in env_id:
        success_mean.append(env.env.success)
        if success_mean[-1]:
            success_rate = 1
        else:
            success_rate = 0

        b = 0
        for i in range(len(success_full)):
            if success_full[i]:
                b += 1
        success_rate_full = b / len(success_full)
        success_full = []

    if "DeepMind" in env_id:
        log_writer.add_scalar("iteration/success_rate_full", success_rate_full, t)
        log_writer.add_scalar("iteration/success_rate", success_rate, t)
        log_writer.add_scalar("iteration/dist_entrance", env.env.dist_entrance, t)
        log_writer.add_scalar("iteration/dist_bottom", env.env.dist_bottom, t)
    log_writer.add_scalar("iteration/reward", opt, t)

    #log_writer.add_scalar("iteration/dist_vec", env.env.dist_vec, t)
    for i in range(len(algorithm.mean)):
        log_writer.add_scalar(f"algorithm_params/mean[{i}]", algorithm.mean[i], t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_mean[{i}]", np.mean(algorithm.C[i]), t)
        log_writer.add_scalar(f"algorithm_params/covariance_matrix_variance[{i}]", np.var(algorithm.C[i]), t)

    return algorithm, env, success_full, success_mean, path, log_writer, opts, t


def policy_kwargs_building(data):
    net_arch = {}
    if data["algo_params"]["policy_type"] == "on_policy":
        net_arch["pi"] = [int(data["algo_params"]["policy_kwargs"]["pi"]), int(data["algo_params"]["policy_kwargs"]["pi"])]
        net_arch["vf"] = [int(data["algo_params"]["policy_kwargs"]["vf"]), int(data["algo_params"]["policy_kwargs"]["vf"])]
        net_arch = [dict(net_arch)]
    elif data["algo_params"]["policy_type"] == "off_policy":
        #net_arch["pi"] = [int(data["algo_params"]["policy_kwargs"]["pi"]), int(data["algo_params"]["policy_kwargs"]["pi"])]
        net_arch["pi"] = [32, 32]
        net_arch["qf"] = [int(data["algo_params"]["policy_kwargs"]["qf"]), int(data["algo_params"]["policy_kwargs"]["qf"])]

    if data["algo_params"]["policy_kwargs"]["activation_fn"] == "tanh":
        activation_fn = th.nn.Tanh
    else:
        activation_fn = th.nn.ReLU
    return dict(activation_fn=activation_fn, net_arch=net_arch)