# Episodic TD3

Incorporating step-based reward feedback into episodic policy.

Our framework is based on TD3 framework from Stable-baselines3 and ProMP framework from Autonomous Learning Robots (ALR) Lab.

Stable-baselines3: https://github.com/DLR-RM/stable-baselines3

Autonomous Learning Robots (ALR) Lab: https://alr.anthropomatik.kit.edu/

## Description for the implementation structure
### Folder config:

The yaml file which stores the parameters. When you want to use our framework, please add an environment configuration into promp_td3.yml

### Folder utils: 
Create the environment, algorithm model, and the callback.

Load the policy parameters from yaml file.

Please see train.py. It is an example for using utils.

### Folder model:
The main structure of our algorithm.

#### Episodic TD3 algorithm part:

| Name                | Description                                            |
|---------------------|--------------------------------------------------------|
| `episodic_td3.py`   | The main algorithm framework of Episodic TD3.          |
| `base_algorithm.py` | The base class of EpisodicTD3 class in episodic_td3.py |


##### Actor ProMP part:

| Name                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `detpmp_model.py`   | Build the ProMP model based on Gaussian Basis Functions.                    |
| `deppmp_wrapper.py` | Wrap the ProMP model to generate reference trajectory based on the weights. |
| `controller.py`     | The controller of ProMP.                                                    |                                                   |            


#### Critic network part:

| Name                | Description                                                           |
|---------------------|-----------------------------------------------------------------------|
| `td3_policy.py`   | Provide the critic networks, without the actor policy neural network. |
| `base_policy.py` | The base policy of TD3 in td3_policy.py.                              |
| `replay_buffer.py`     | The Replay Buffer with timestep information                                          |                                                   |            

## How to use

TODO
