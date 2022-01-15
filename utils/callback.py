import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def callback_function(data: dict):
    if 'VecNormalize' in data["env_params"]['wrapper']:
        callback_class = 'VecNormalizeCallback'
    else:
        callback_class = 'DummyCallback'
    return callback_class

class ALRBallInACupCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(ALRBallInACupCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        last_dist = np.mean([self.model.env.venv.envs[i].last_dist \
                             for i in range(len(self.model.env.venv.envs))
                             if self.model.env.venv.envs[i].last_dist != 0])
        last_dist_final = np.mean([self.model.env.venv.envs[i].last_dist_final \
                                   for i in range(len(self.model.env.venv.envs))
                                   if self.model.env.venv.envs[i].last_dist_final != 0])
        total_dist= np.mean([self.model.env.venv.envs[i].total_dist \
                             for i in range(len(self.model.env.venv.envs))])
        total_dist_final = np.mean([self.model.env.venv.envs[i].total_dist_final \
                                   for i in range(len(self.model.env.venv.envs))])
        min_dist = np.mean([self.model.env.venv.envs[i].min_dist \
                            for i in range(len(self.model.env.venv.envs))])
        min_dist_final = np.mean([self.model.env.venv.envs[i].min_dist_final \
                                  for i in range(len(self.model.env.venv.envs))])
        step = np.mean([self.model.env.venv.envs[i].step_record \
                        for i in range(len(self.model.env.venv.envs))])
        self.logger.record('reward/step', step)
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        return True



class ALRReacherCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(ALRBallInACupCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        '''
        last_dist = np.mean([self.model.env.venv.envs[i].last_dist \
                             for i in range(len(self.model.env.venv.envs))
                             if self.model.env.venv.envs[i].last_dist != 0])
        last_dist_final = np.mean([self.model.env.venv.envs[i].last_dist_final \
                                   for i in range(len(self.model.env.venv.envs))
                                   if self.model.env.venv.envs[i].last_dist_final != 0])
        total_dist= np.mean([self.model.env.venv.envs[i].total_dist \
                             for i in range(len(self.model.env.venv.envs))])
        total_dist_final = np.mean([self.model.env.venv.envs[i].total_dist_final \
                                   for i in range(len(self.model.env.venv.envs))])
        min_dist = np.mean([self.model.env.venv.envs[i].min_dist \
                            for i in range(len(self.model.env.venv.envs))])
        min_dist_final = np.mean([self.model.env.venv.envs[i].min_dist_final \
                                  for i in range(len(self.model.env.venv.envs))])
        step = np.mean([self.model.env.venv.envs[i].step_record \
                        for i in range(len(self.model.env.venv.envs))])
        self.logger.record('reward/step', step)
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        '''
        step = np.mean([self.model.env.venv.envs[i].step_record \
                        for i in range(len(self.model.env.venv.envs))])
        self.logger.record('reward/reward', step)
        return True

class DMbicCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(DMbicCallback, self).__init__(verbose)
        self.success_rate = 0
        
    def reset(self):
        self.success_rate = 0
        return super().reset()

    def _on_step(self) -> bool:

        success_rate = 0
        for i in range(len(self.model.env.venv.envs)):
            #print(self.model.env.venv.envs[i].success_final)
            success_rate += int(self.model.env.venv.envs[i].success)
            #print("callback", i, self.model.env.venv.envs[i].success)
            #print("success_rate", success_rate)
            #if self.model.env.venv.envs[i].success_final:
            #    self.success_rate += 1
            #    print("success_rate", self.success_rate)

        success_rate = success_rate / len(self.model.env.venv.envs)
        #success_rates.append(success_rate)
        self.logger.record('reward/success_rate', success_rate)
        return True


class DummyCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(DummyCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        last_dist = np.mean([self.model.env.envs[i].last_dist \
                             for i in range(len(self.model.env.envs))
                             if self.model.env.envs[i].last_dist != 0])
        last_dist_final = np.mean([self.model.env.envs[i].last_dist_final \
                                   for i in range(len(self.model.env.envs))
                                   if self.model.env.envs[i].last_dist_final != 0])
        total_dist= np.mean([self.model.env.envs[i].total_dist \
                             for i in range(len(self.model.env.envs))])
        total_dist_final = np.mean([self.model.env.envs[i].total_dist_final \
                                   for i in range(len(self.model.env.envs))])
        min_dist = np.mean([self.model.env.envs[i].min_dist \
                            for i in range(len(self.model.env.envs))])
        min_dist_final = np.mean([self.model.env.envs[i].min_dist_final \
                                  for i in range(len(self.model.env.envs))])
        step = np.mean([self.model.env.envs[i].step_record \
                        for i in range(len(self.model.env.envs))])
        self.logger.record('reward/step', step)
        self.logger.record('reward/last_dist', last_dist)
        self.logger.record('reward/last_dist_final', last_dist_final)
        self.logger.record('reward/total_dist', total_dist)
        self.logger.record('reward/total_dist_final', total_dist_final)
        self.logger.record('reward/min_dist', min_dist)
        self.logger.record('reward/min_dist_final', min_dist_final)
        return True