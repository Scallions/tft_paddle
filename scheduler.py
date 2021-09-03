import math
from paddle.optimizer.lr import LRScheduler

class CosineAnnealingDecay(LRScheduler):

    def __init__(self,
                 learning_rate,
                 T_period,
                 restarts=None,
                 weights=None,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        self.T_period = T_period
        self.T_max = self.T_period[0]
        self.eta_min = float(eta_min)
        self.restarts = restarts if restarts else [0]
        self.restarts = [v + 1 for v in self.restarts]
        self.restart_weights = weights if weights else [1]
        self.last_restart = 0
        super(CosineAnnealingDecay, self).__init__(learning_rate, last_epoch,
                                                   verbose)
    def get_lr(self):
        if self.last_epoch == 0:
            return self.base_lr

        elif self.last_epoch in self.restarts:
            self.last_restart = self.last_epoch
            self.T_max = self.T_period[self.restarts.index(self.last_epoch) + 1]
            weight = self.restart_weights[self.restarts.index(self.last_epoch)]
            return self.base_lr * weight
            #return self.base_lr

        elif (self.last_epoch - self.last_restart - 1 - self.T_max) % (2 * self.T_max) == 0:
            return self.last_lr + (self.base_lr - self.eta_min) * (1 - math.cos(math.pi/self.T_max)) / 2

        return (1 + math.cos(math.pi * (self.last_epoch - self.last_restart) / self.T_max)) / (1 + math.cos(math.pi * ((self.last_epoch - self.last_restart) - 1) / self.T_max)) * (self.last_lr - self.eta_min) + self.eta_min

    def _get_closed_form_lr(self):
        return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
            math.pi * self.last_epoch / self.T_max)) / 2