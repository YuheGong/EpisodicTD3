import pandas as pd
import numpy as np
from scipy.interpolate import make_interp_spline
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm
import matplotlib.pyplot as plt


def dict_building(folder, name, num):
    path = "./data/" \
           + folder + "/" + name
    plotdict = {'df{}'.format(q): pd.read_csv(path + "_{}".format(q) + "/data.csv", nrows=num) for q in
                range(1, 2)}

    # plotdict = plotdict[0:5001]
    # plotdict.update({'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
    #                 range(10, 20)})
    return plotdict


def dict_building_self(folder, name):
    path = "./data/" \
           + folder + "/" + name
    plotdict = {'df{}'.format(q): pd.read_csv(path + "_{}".format(q) + "/data" + ".csv") for q in
                range(1, 21)}
    # plotdict.update({'df{}'.format(q): pd.read_csv(path + "/rep_{}".format(q) + "/rep_{}".format(q) + ".csv") for q in
    #                 range(10, 20)})
    return plotdict


def plot_function(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)
    X_Y_Spline = make_interp_spline(plot_samples.reshape(-1), plot_mean.reshape(-1))
    X = np.linspace(plot_samples.min(), plot_samples.max(), 50)
    Y = X_Y_Spline(X)

    Z_Y_Spline = make_interp_spline(plot_samples.reshape(-1), var.reshape(-1))
    # X = np.linspace(plot_samples.min(), plot_samples.max(), 50)
    Z = Z_Y_Spline(X)
    # plt.plot(plot_samples, plot_mean, label=algo.upper())
    #plt.plot(X, Y, label=algo.upper())
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)
    plt.plot(plot_samples, plot_mean, label=algo.upper())
    plt.fill_between(plot_samples, plot_mean - var, plot_mean + var, alpha=0.2)


def plot_function_pt3(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)
    X_Y_Spline = make_interp_spline(plot_samples.reshape(-1), plot_mean.reshape(-1))
    X = np.linspace(plot_samples.min(), plot_samples.max(), 50)
    Y = X_Y_Spline(X)

    Z_Y_Spline = make_interp_spline(plot_samples.reshape(-1), var.reshape(-1))
    # X = np.linspace(plot_samples.min(), plot_samples.max(), 50)
    Z = Z_Y_Spline(X)
    # print("z", Z)
    # print("z",  var)
    # assert 1==123
    # plt.plot(plot_samples, plot_mean, label=algo.upper())
    #plt.plot(X, Y, label="Episodic TD3")
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)
    plt.plot(plot_samples, plot_mean, label="Episodic TD3")
    plt.fill_between(plot_samples, plot_mean - var, plot_mean + var, alpha=0.2)


def plot_function_promp(result, algo):
    plot_mean = result['eval/mean_reward'].reshape(-1)
    plot_samples = np.array(result['step']).reshape(-1)
    var = np.array(result['var']).reshape(-1)
    X_Y_Spline = make_interp_spline(plot_samples.reshape(-1), plot_mean.reshape(-1))
    X = np.linspace(plot_samples.min(), plot_samples.max(), 50000)
    Y = X_Y_Spline(X)

    Z_Y_Spline = make_interp_spline(plot_samples.reshape(-1), var.reshape(-1))
    # X = np.linspace(plot_samples.min(), plot_samples.max(), 50)
    Z = Z_Y_Spline(X)
    #plt.plot(X, Y, label="ProMP")
    #plt.fill_between(X, Y - Z, Y + Z, alpha=0.2)
    plt.plot(plot_samples, plot_mean, label="ProMP")
    plt.fill_between(plot_samples, plot_mean - var, plot_mean + var, alpha=0.2)


def suc_rate_value(plotdict, value):
    # plotdict = plotdict[0:5001]
    success_rate_full = []
    # print("plotdict",plotdict)
    for k in plotdict.items():
        success_rate_full.append(k[1][value])

    success_rate_value = np.array(success_rate_full)
    return success_rate_value


def label_name(name):
    if "DMP" in name:
        a = "DMP"
    elif "ProMP" in name:
        a = "ProMP"
    else:
        a = 'should write'
    if name[-1] == "0":
        b = 'exp'
    elif name[-1] == "1":
        b = 'quad'
    elif name[-1] == "2":
        b = 'log'
    if "Dense" in name:
        c = "dense"
    else:
        c = "sparse"
    return a + ' - ' + b + ' - ' + c


def csv_save(folder, name, algo, foler_num):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(1,6):
        path = "./" \
               + folder + "/" + name
        in_path = path + '_' + f'{i}' + '/' + algo + '_1'
        ex_path = path + '_' + f'_{i}' + '/' + "eval_reward_mean.csv"
        event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        event_data.Reload()
        tags = event_data.Tags()

        keys = event_data.scalars.Keys()  # get all tags,save in a list
        for hist in tags['scalars']:
            if hist == 'eval/mean_reward':
                histograms = event_data.Scalars(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))

        #for k in range(1, rewards[i-1].shape[0]):
        #    if rewards[i-1][k] < rewards[i-1][k-1]:
        #        rewards[i-1][k] = rewards[i-1][k-1]

        a = 1
    rewards = np.array(rewards)
    var = np.std(rewards, axis=0)
    rewards = rewards.mean(axis=0)
    steps = np.array(steps)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


def csv_save_promp(folder, name, algo, foler_num):
    # save csv file
    steps = []
    rewards = []
    result = {}
    for i in range(1,6):
        path = "./" \
               + folder + "/" + name
        in_path = path + '_' + f'{i}' + '/' + algo #+ '_1'
        ex_path = path + '_' + f'_{i}' + '/' + "eval_reward_mean.csv"
        event_data = event_accumulator.EventAccumulator(in_path)  # a python interface for loading Event data
        event_data.Reload()  # synchronously loads all of the data written so far b
        # print(event_data.Tags())  # print all tags
        event_data.Reload()
        tags = event_data.Tags()

        keys = event_data.scalars.Keys()  # get all tags,save in a list
        for hist in tags['scalars']:
            if hist == 'eval/mean_reward':
                histograms = event_data.Scalars(hist)
                rewards.append(np.array(
                    [np.array(h.value) for
                     h in histograms]))
                steps.append(np.array(
                    [np.array(h.step) for
                     h in histograms]))

        #for k in range(1, rewards[i - 1].shape[0]):
        #    if rewards[i - 1][k] < rewards[i - 1][k - 1]:
        #        rewards[i - 1][k] = rewards[i - 1][k - 1]

    rewards = np.array(rewards)
    var = np.std(rewards, axis=0)
    rewards = rewards.mean(axis=0)
    steps = np.array(steps)
    steps = steps.mean(axis=0)

    result['eval/mean_reward'] = rewards
    result['step'] = steps
    result['var'] = var
    return result


folder = "data"
value = "eval/mean_reward"
# algo = "sac"
# algo = "ProMP"

folder_num = 12  # 12

env = "dmcCheetahDense-v0"
#env = "dmcWalkerDense-v0"
#env_promp = "dmcWalkerDenseProMP-v0"
env_promp = "dmcCheetahDenseProMP-v0"
# env = "FetchReacher-v0"
# folder = "forthweek"
# value = "reward"

for v in range(1):
    # name = "DeepMindBallInCup" + algo + "-v{}".format(v)
    # name = algo + "/" + env  #+ algo + "-v{}".format(v)
    # result = csv_save(folder, name, algo.upper(), folder_num)
    # plotdict = dict_building(folder, name, num=325)
    # success_rate_value = suc_rate_value(plotdict,value)
    # plot_function(result,algo)

    algo = "td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, 'TD3', folder_num)
    # plotdict = dict_building(folder, name,num=1000)
    # success_rate_value = suc_rate_value(plotdict, value)
    plot_function(result, algo)

    algo = "episodic_td3"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "run", folder_num)
    # plotdict = dict_building(folder, name,num=1000)
    # success_rate_value = suc_rate_value(plotdict, value)
    plot_function_pt3(result, algo)

    algo = "sac"
    name = algo + "/" + env  # + algo + "-v{}".format(v)
    result = csv_save(folder, name, "SAC", folder_num)
    # plotdict = dict_building(folder, name,num=1000)
    # success_rate_value = suc_rate_value(plotdict, value)
    plot_function(result, algo)

    algo = "promp"
    name = algo + "/" + env_promp  # + algo + "-v{}".format(v)
    result = csv_save_promp(folder, name, "", folder_num)
    # result["step"] = result["step"] /1250 * 1000
    # plotdict = dict_building(folder, name,num=1000)
    # success_rate_value = suc_rate_value(plotdict, value)
    plot_function_promp(result, algo)

# csv_save(folder, name)
# plt.title("ALR Reacher - Line trajectory")
plt.title("Deep Mind Walker")
plt.xlabel("timesteps")
plt.ylabel("rewards")
plt.ylim(ymin=0)
# plt.title("ALRReacher-v3")
# plt.ylim(ymin=-100)
plt.ylim(ymax=200)
plt.legend()
plt.show()
plt.savefig("latex/alr5.png")

import tikzplotlib

# tikzplotlib.save("latex/alr3.tex")
tikzplotlib.save("latex/alr5.tex")



