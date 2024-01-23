import numpy as np
from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter

from pymodaq_plugins_mockexamples.hardware.pinem_simulator import PinemGenerator
from pymodaq_plugins_mockexamples.daq_viewer_plugins.plugins_1D.daq_1Dviewer_Pinem import DAQ_1DViewer_Pinem

class DAQ_1DViewer_PinemAnalysis(DAQ_1DViewer_Pinem):
    """ Instrument plugin class for a 1D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    Attributes:
    -----------
    controller: object
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.

    """

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

        g1, g2, theta = self.my_pinem_algo(data_array)

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



if __name__ == '__main__':
    main(__file__)
