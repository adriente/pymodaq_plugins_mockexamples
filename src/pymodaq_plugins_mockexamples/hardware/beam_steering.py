# -*- coding: utf-8 -*-
"""
Created the 24/10/2022

@author: Sebastien Weber
"""
import numpy as np
import pymodaq.utils.math_utils as mutils

from pymodaq_plugins_mockexamples.hardware.camera_wrapper import Camera
from pymodaq_plugins_mockexamples.hardware.wrapper import ActuatorWrapperWithTauMultiAxes


class BeamSteeringActuators(ActuatorWrapperWithTauMultiAxes):
    axes = Camera.axes + ['Power']
    _units = Camera.units + ['W']
    _epsilon = 0.01
    _tau = 0.3  # s

    def __init__(self):
        super().__init__()

        self._current_values = [128, 128, 0,  1]


class BeamSteering():
    _tau = BeamSteeringActuators._tau

    def __init__(self):

        self.actuators = BeamSteeringActuators()
        self.camera = Camera()
        self.camera.fringes = False

    @property
    def tau(self):
        """
        fetch the characteristic decay time in s
        Returns
        -------
        float: the current characteristic decay time value

        """
        return self.actuators.tau

    @tau.setter
    def tau(self, value: float):
        """
        Set the characteristic decay time value in s
        Parameters
        ----------
        value: (float) a strictly positive characteristic decay time
        """
        self.actuators.tau = value

    def move_at(self, value: float, axis: str):
        """
        """
        if axis in BeamSteeringActuators.axes:
            self.actuators.move_at(value, axis)
        if axis in Camera.axes:
            self.camera.set_value(axis, value)

    def stop(self, axis: str):
        self.actuators.stop(axis)

    def get_value(self, axis: str):
        """
        Get the current actuator value
        Returns
        -------
        float: The current value
        """
        return self.actuators.get_value(axis)

    def get_camera_data(self) -> np.ndarray:
        return self.camera.get_data() * self.actuators.get_value('Power')

    def get_photodiode_data(self) -> float:
        
        idx = int(Camera.Nx / 2)
        idy = int(Camera.Ny / 2)
        width = 20
        
        return np.mean(
            self.camera.get_data()[
               int(idy-width/2): int(idy+width/2),
               int(idx-width/2): int(idx+width/2)]) * self.actuators.get_value('Power')
