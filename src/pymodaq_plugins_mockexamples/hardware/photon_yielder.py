# -*- coding: utf-8 -*-
"""
Created the 30/10/2023

@author: Sebastien Weber
"""
from pathlib import Path

import numpy as np


import numpy as np

here = Path(__file__).parent.parent
data_memory_map = np.load(here.joinpath('resources/KXe_000203_raw.npy'), mmap_mode='r')


class Photon:

    def __init__(self, photon_array: np.ndarray):
        self.index = int(photon_array[0])
        self.time_stamp = float(photon_array[4])
        self.x_pos = int(photon_array[2])
        self.y_pos = int(photon_array[3])
        self.intensity = int(photon_array[5])

    def to_positions_intensity(self):
        return np.array([self.x_pos, self.y_pos, self.intensity])

    def __repr__(self):
        return f'Photon event {self.index}: x:{self.x_pos}, y: {self.y_pos},' \
               f' time: {self.time_stamp}, intensity: {self.intensity}'


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
    from pathlib import Path
    from qtpy import QtCore
    from pymodaq.utils.h5modules.saving import H5SaverLowLevel
    from pymodaq.utils.daq_utils import ThreadCommand
    from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport, DataRaw, DataCalculated, DataWithAxes
    from pymodaq.utils.h5modules.data_saving import DataEnlargeableSaver, DataToExportEnlargeableSaver, DataLoader, DataSaverLoader

    here = Path(r'C:\Users\weber\Labo\Programmes Python\PyMoDAQ_Git\pymodaq_plugins_folder\pymodaq_plugins_mockexamples\src\pymodaq_plugins_mockexamples\resources')

    addhoc_file_path = here.joinpath('photons.h5')
    h5temp = H5SaverLowLevel()
    h5temp.init_file(file_name=addhoc_file_path, new_file=True)
    h5temp.get_set_group('/RawData', 'myphotons')

    saver = DataSaverLoader(h5temp)

    data = DataWithAxes('timepix', source='raw', distribution='spread', nav_indexes=(0,),
                        data=[data_memory_map[:, 5], data_memory_map[:, 2], data_memory_map[:, 3]],
                        axes=[Axis('tof', 's', data=data_memory_map[:, 4])])
    saver.add_data('/RawData/myphotons', data, save_axes=True)
    h5temp.close_file()
    photon = PhotonYielder()
    ind = 0
    for ind in range(100):
        print(photon.grab())


    pass
