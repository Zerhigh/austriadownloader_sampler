import numpy as np


class ImageConfig:
    def __init__(self, pixel_size=1.6, shape=(4, 1024, 1024)):
        self.pixel_size = pixel_size
        self.shape = shape

        # corine defined raster size
        self.raster_size = 100

    @property
    def window_size(self):
        width = (self.pixel_size * self.shape[1]) // 2
        # reduce window size by 1 to account for sampled pixel width (from 0 to 100m in one direction)
        return int(np.floor(width / self.raster_size)) - 1
