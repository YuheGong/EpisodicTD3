# Episodic TD3

Incorporating step-based reward feedback into episodic policy.

Our framework is based on TD3 framework from Stable-baselines3 and ProMP framework from Autonomous Learning Robots (ALR) Lab.

Stable-baselines3: https://github.com/DLR-RM/stable-baselines3

Autonomous Learning Robots (ALR) Lab: https://alr.anthropomatik.kit.edu/

## Folder utils: 
Create the environment, model. and the callback.

Load the policy parameters from yaml file 

## Folder model:
The main structure of our algorithm.

### Episodic TD3:

promp_td3.py: the main algorithm framework.

base_algorithm.py: the base class of Episodic TD3 in promp_td3.py 

#### Actor ProMP:

detpmp_model.py: build the ProMP model based on Gaussian Basis Functions.

deppmp_wrapper.py: wrap the ProMP model to generate reference trajectory based on the weights.

controller.py: the controller used for ProMP

#### Critic Model :

td3_policy.py: provide the critic networks, without the actor policy neural network

base_policy.py: the base policy of TD3 in td3_policy.py

replay_buffer.py: the Replay Buffer with timestep information
