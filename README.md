# Episodic TD3

Introduce the movement primitives into actor-critic algorithms

Our framework is based on TD3 framework from Stable-baselines3 and ProMP framework from Autonomous Learning Robots (ALR) Lab.

Stable-baselines3: https://github.com/DLR-RM/stable-baselines3

Autonomous Learning Robots (ALR) Lab: https://alr.anthropomatik.kit.edu/

## Description of the implementation structure
| Name          | Description                                      |
|---------------|--------------------------------------------------|
| `train.py`    | An example for training an environment.          |
| `enjoy.py`    | An example for enjoying an environment.          |
| `continue.py` | An example for continue training an environment. |



### FOLDER config:

The yaml file which stores the parameters. When you want to use our framework, please add an environment configuration into context.yml or non_context.yml

### FOLDER utils: 
Create the environment, algorithm model, and the callback.

Load the hyperparameters from yaml file.

Please see train.py. It is an example for using utils.

### FOLDER model:
The main structure of our algorithm.

#### Training and Sampling:

| Name                | Description                                            |
|---------------------|--------------------------------------------------------|
| `episodic_td3.py`   | The main algorithm framework of Episodic TD3.          |
| `base_algorithm.py` | The base class of EpisodicTD3 class. |


##### Actor:

| Name                | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `detpmp_model.py`   | Build the ProMP model based on Gaussian Basis Functions.                    |
| `deppmp_wrapper.py` | Wrap the ProMP model to generate reference trajectory based on the weights. |
| `controller.py`     | The controller of ProMP.                                                    |                                                   |            


#### Critic:

| Name                | Description                                                           |
|---------------------|-----------------------------------------------------------------------|
| `td3_policy.py`   | Provide the critic networks, contextual actor network. |
| `base_policy.py` | The base class of td3_policy.py.                              |
| `replay_buffer.py`     | The Replay Buffer with normalized timestep information                                          |                                                   |            
