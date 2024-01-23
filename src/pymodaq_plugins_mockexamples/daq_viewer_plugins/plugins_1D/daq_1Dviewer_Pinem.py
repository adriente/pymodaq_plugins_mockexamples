import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_mockexamples.hardware.pinem_simulator import PinemGenerator

class DAQ_1DViewer_Pinem(DAQ_Viewer_base):
    """ Instrument plugin class for a 1D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """
    params = comon_parameters+[
        {'title': 'g1', 'name': 'g1', 'type': 'slide', 'value': 1, 'default': 1, 'min': 0,
         'max': 5, 'subtype': 'linear'},
        # elements to be added here as dicts in order to control your custom stage
        ############
        ]

    def ini_attributes(self):
        self.controller: PinemGenerator = None
        self.x_axis = None

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        if param.name() == "g1":
            self.controller.g1 = param.value()

    def ini_detector(self, controller=None):
        """Detector communication initialization

        Parameters
        ----------
        controller: (object)
            custom object of a PyMoDAQ plugin (Slave case). None if only one actuator/detector by controller
            (Master case)

        Returns
        -------
        info: str
        initialized: bool
            False if initialization failed otherwise True
        """
        self.ini_detector_init(old_controller=controller,
                               new_controller=PinemGenerator(1024, 'Gaussian'))

        info = "Whatever info you want to log"
        initialized = True
        return info, initialized

    def close(self):
        """Terminate the communication protocol"""
        ...

    def grab_data(self, Naverage=1, **kwargs):
        """Start a grab from the detector

        Parameters
        ----------
        Naverage: int
            Number of hardware averaging (if hardware averaging is possible, self.hardware_averaging should be set to
            True in class preamble and you should code this implementation)
        kwargs: dict
            others optionals arguments
        """
        data_array = self.controller.gen_data()
        axis = Axis('energy', data=self.controller.x)
        self.dte_signal.emit(DataToExport('Pinem',
                                  data=[
                                        DataFromPlugins(name='Constants',
                                                        data=[np.array([self.controller.g1]),
                                                              np.array([self.controller.g2]),
                                                              np.array([self.controller.theta])],
                                                        dim='Data0D', labels=['g1', 'g2', 'theta']),
                                      DataFromPlugins(name='Spectrum', data=[data_array],
                                                      dim='Data1D', labels=['Spectrum'],
                                                      axes=[axis]),
                                        ]))

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        return ''


if __name__ == '__main__':
    main(__file__)
