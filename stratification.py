
class Stratification:
    def __init__(self, clc_distribution, window_treshhold, num_samples='max'):
        self.clc_distribution = clc_distribution
        self.window_treshhold = window_treshhold
        self.num_samples = num_samples

    @property
    def clc_filtered(self):
        return {k: v for k, v in self.clc_distribution.items() if v > 0}

