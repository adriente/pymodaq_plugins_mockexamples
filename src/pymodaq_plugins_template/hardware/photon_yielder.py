# -*- coding: utf-8 -*-
"""
Created the 30/10/2023

@author: Sebastien Weber
"""


import numpy as np
data_memory_map = np.load('../resources/KXe_000203_raw.npy', mmap_mode='r')


class Photon:

    def __init__(self, photon_array: np.ndarray):
        self.index = int(photon_array[0])
        self.time_stamp = int(photon_array[1])
        self.x_pos = int(photon_array[2])
        self.y_pos = int(photon_array[3])
        self.intensity = int(photon_array[4])

    def __repr__(self):
        return f'Photon event {self.index}: x:{self.x_pos}, y: {self.y_pos}, time: {self.time_stamp}'


class PhotonYielder:
    ind_grabed = -1

    def __init__(self):
        self._photon_grabber = self._grabber()

    def _grabber(self):
        while self.ind_grabed < data_memory_map.shape[0]:
            self.ind_grabed += 1
            yield data_memory_map[self.ind_grabed, ...]

    def grab(self) -> Photon:
        return Photon(next(self._photon_grabber))


if __name__ == '__main__':
    photon = PhotonYielder()
    ind = 0
    for ind in range(100):
        print(photon.grab())
