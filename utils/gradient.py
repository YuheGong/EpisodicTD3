import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class GradientDescent():
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