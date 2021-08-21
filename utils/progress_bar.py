# -*- coding: utf-8 -*-
# ---------------------

import math
from datetime import datetime


class ProgressBar(object):
    """
    Utility class for the management of progress bars showing training progress in the form
    "[<date>] Epoch <epoch_number>.<step_number> │<progres_bar>│ <completion_percentage>"
    """


    @property
    def progress(self):
        return (self.step + 1) / self.max_step


    def __init__(self, max_step, max_epoch, current_epoch=0):
        self.max_step = max_step
        self.max_epoch = max_epoch
        self.current_epoch = current_epoch
        self.step = 0


    def inc(self):
        """
        Increase the progress bar value by one unit
        """
        self.step = self.step + 1
        if self.step == self.max_step:
            self.step = 0
            self.current_epoch = self.current_epoch + 1


    def __str__(self):
        value = int(round(self.progress * 50))
        date = datetime.now().strftime("%b-%d@%H:%M").lower()
        progress_bar = ('█' * value + ('┈' * (50 - value)))
        return '\r[{}] Epoch {:0{e}d}.{:0{s}d}: │{}│ {:6.2f}%'.format(
            date, self.current_epoch, self.step + 1,
            progress_bar, 100 * self.progress,
            e=math.ceil(math.log10(self.max_epoch)),
            s=math.ceil(math.log10(self.max_step + 1)),
        )
