# -*- coding: utf-8 -*-
"""
Created the 04/11/2023

@author: Constant Schouder
"""
import re
from typing import List

import numpy as np

from pymodaq.utils import math_utils as mutils
from pymodaq.utils.scanner.scan_factory import ScannerFactory
from pymodaq.utils.scanner.scanners._1d_scanners import Scan1DBase, DataDistribution


class Scan1DSparse(Scan1DBase): #Matlab syntax class for easy scan creation
    ''' Syntax goes as start:step:stop, such as 0:0.2:1 will give [0 0.2 0.4 0.6 0.8 1] or single entry 0 will give 0
      Separate entries with comma or new line, such 0:0.2:1,5 will give [0 0.2 0.4 0.6 0.8 1 5]
    '''
    scan_type = 'Scan1D'
    scan_subtype = 'SparseExample'
    params = [
        {'title': 'Parsed string:', 'name': 'parsed_string', 'type': 'str', 'value': '0:0.1:1', }
        ]
    n_axes = 1
    distribution = DataDistribution['spread']

    def __init__(self, actuators: List = None, **_ignored):
        super().__init__(actuators=actuators)
        self.settings.child('parsed_string').setOpts(tip=self.__doc__)

    def set_scan(self):
        range_strings = re.findall("[^,\s]+", self.settings['parsed_string'])
        series = np.asarray([])
        for range_string in range_strings:
            number_strings = re.findall("[^:]+", range_string)  # Extract the numbers by splitting on :.
            if len(number_strings) == 3:  # 3 Numbers specify a range
                start, step, stop = [float(number) for number in number_strings]
                this_range = mutils.linspace_step(start, stop, step)
            elif len(number_strings) == 1:  # 1 number just specifies a single number
                this_range = np.asarray([float(number_strings[0])])
            series = np.concatenate((series, this_range))
        self.positions = series
        self.get_info_from_positions(self.positions)

    def set_settings_titles(self):
        if len(self.actuators) == 1:
            self.settings.child('start').setOpts(title=f'{self.actuators[0].title} start:')

    def evaluate_steps(self) -> int:
        """Quick evaluation of the number of steps to stop the calculation if the evaluation os above the
        configured limit"""
        self.set_scan()  # no possible quick evaluation, easiest to process it
        return self.n_steps

    def set_settings_titles(self):
        if len(self.actuators) == 1:
            self.settings.child('parsed_string').setOpts(title=f'{self.actuators[0].title} Parsed string:')
