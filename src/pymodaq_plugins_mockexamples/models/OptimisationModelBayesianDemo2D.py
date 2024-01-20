from typing import List, Union, TYPE_CHECKING
from pathlib import Path

import numpy as np
from qtpy import QtWidgets, QtCore

from pymodaq_plugins_optimisation.utils import OptimisationModelGeneric, DataToActuatorOpti, OptimisationAlgorithm
from pymodaq_plugins_optimisation.hardware.gershberg_saxton import GBSAX
from pymodaq.utils import gui_utils as gutils
from pymodaq.utils.plotting.data_viewers import Viewer2D, ViewersEnum
from pymodaq.utils.logger import set_logger, get_module_name
from pymodaq.utils.data import DataToExport, DataActuator, DataRaw
from pymodaq_plugins_optimisation.algorithms.bayesian_opti import Algorithm


if TYPE_CHECKING:
    from pymodaq_plugins_optimisation.extensions.optimisation import Optimisation

logger = set_logger(get_module_name(__file__))


class OptimisationModelBayesianDemo2D(OptimisationModelGeneric):

    optimisation_algorithm: Algorithm = None

    actuators_name = ["Xaxis", "Yaxis"]
    detectors_name = ["ComplexData"]
    observables_dim = [ViewersEnum('Data0D')]

    params = [
        {'title': 'Ini. State', 'name': 'ini_random', 'type': 'int', 'value': 10},
        {'title': 'refresh function', 'name': 'refresh', 'type': 'bool_push', 'label': 'refresh'},
        {'title': 'X', 'name': 'x', 'type': 'group', 'children':[
            {'title': 'xmin', 'name': 'min', 'type': 'float', 'value': -5},
            {'title': 'xmax', 'name': 'max', 'type': 'float', 'value': 5},
        ]},

        {'title': 'Y', 'name': 'y', 'type': 'group', 'children': [
            {'title': 'ymin', 'name': 'min', 'type': 'float', 'value': -5},
            {'title': 'ymax', 'name': 'max', 'type': 'float', 'value': 5},
        ]}

    ]

    def __init__(self, optimisation_controller: 'Optimisation'):
        super().__init__(optimisation_controller)

        self.problem = None
        self.other_detectors: List[str] = []

        target_dock = gutils.Dock('Target')
        widget_target = QtWidgets.QWidget()
        target_dock.addWidget(widget_target)
        self.optimisation_controller.dockarea.addDock(target_dock, 'bottom',
                                                      self.optimisation_controller.docks['settings'])
        self.viewer_target = Viewer2D(widget_target)

    def update_settings(self, param):
        """
        Get a parameter instance whose value has been modified by a user on the UI
        """
        if param.name() == 'refresh':
            self.update_function()

    def ini_model(self):
        super().ini_models()
        self.update_function()

    def ini_algo(self):
        self.optimisation_algorithm = Algorithm(ini_random=self.settings['ini_random'],
                                                bounds={'x': (self.settings['x', 'min'],
                                                              self.settings['x', 'max']),
                                                        'y': (self.settings['y', 'min'],
                                                              self.settings['y', 'max']),
                                                        })

    def update_function(self):
        self.ini_algo()
        self.modules_manager.detectors[0].controller.ini_random_structures()
        dwa = self.modules_manager.detectors[0].controller.get_random_hypergaussian_datagrid()
        self.viewer_target.show_data(dwa)

    def convert_input(self, measurements: DataToExport) -> DataToExport:
        """
        Convert the measurements in the units to be fed to the Optimisation Controller
        Parameters
        ----------
        measurements: DataToExport
            data object exported from the detectors from which the model extract a value of the same units as
            the setpoint

        Returns
        -------
        DataToExport

        """
        return measurements

    def convert_output(self, outputs: List[np.ndarray]) -> DataToActuatorOpti:
        """
        Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: list of numpy ndarray
            output value from the controller from which the model extract a value of the same units as the actuators

        Returns
        -------
        DataToActuatorOpti: derived from DataToExport. Contains value to be fed to the actuators with a mode
            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        try:
            xs, ys = self.viewer_target.view.unscale_axis(float(self.optimisation_algorithm.best_individual[0]),
                                                          float(self.optimisation_algorithm.best_individual[1]))
            self.viewer_target.double_clicked(xs, ys)
        except KeyError:
            pass
        return DataToActuatorOpti('outputs', mode='abs', data=[DataActuator(self.actuators_name[0],
                                                                            data=float(outputs[0])),
                                                               DataActuator(self.actuators_name[1],
                                                                            data=float(outputs[1]))])


if __name__ == '__main__':
    pass


