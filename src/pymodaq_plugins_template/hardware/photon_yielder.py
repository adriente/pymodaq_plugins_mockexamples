# -*- coding: utf-8 -*-
"""
Created the 30/10/2023

@author: Sebastien Weber
"""


import numpy as np
data_memory_map = np.load('../resources/KXe_000203_raw.npy', mmap_mode='r')


class PhotonYielder:
    ind_grabed = -1

    def grabber(self):
        while self.ind_grabed < data_memory_map.shape[0]:
            self.ind_grabed += 1
            yield data_memory_map[self.ind_grabed, ...]


if __name__ == '__main__':
    photon = PhotonYielder()
    photon_grabber = photon.grabber()
    ind = 0
    for ind in range(100):
        print(next(photon_grabber))
