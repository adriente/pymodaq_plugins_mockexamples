from typing import List
from qtpy.QtWidgets import QWidget

import numpy as np

from pymodaq.extensions.bayesian.utils import BayesianModelDefault
from pymodaq.utils import gui_utils as gutils
from pymodaq.utils.plotting.data_viewers import Viewer1D, Viewer2D

from pymodaq.utils.data import DataToExport, DataActuator, DataToActuators


class BayesianModelMock(BayesianModelDefault):

    def ini_model(self):


        dock_mock = gutils.Dock('Mock Data')
        dock_widget = QWidget()
        dock_mock.addWidget(dock_widget)
        self.optimisation_controller.dockarea.addDock(dock_mock, 'bottom')

        controller = self.modules_manager.get_mod_from_name('ComplexData').controller
        dwa = controller.get_data_grid()
        if controller.signal_type == 'Lorentzian':
            self.viewer_mock = Viewer1D(dock_widget)
        else:
            self.viewer_mock = Viewer2D(dock_widget)

        self.viewer_mock.show_data(dwa)
        self.viewer_mock.get_action('crosshair').trigger()

    def convert_output(self, outputs: List[np.ndarray], best_individual=None) -> DataToActuators:
        """ Convert the output of the Optimisation Controller in units to be fed into the actuators
        Parameters
        ----------
        outputs: list of numpy ndarray
            output value from the controller from which the model extract a value of the same units
            as the actuators
        best_individual: np.ndarray
            the coordinates of the best individual so far

        Returns
        -------
        DataToActuators: derived from DataToExport. Contains value to be fed to the actuators
        with a mode            attribute, either 'rel' for relative or 'abs' for absolute.

        """
        try:
            if best_individual is not None:
                pos_list = [float(coord) for coord in best_individual]
                if isinstance(self.viewer_mock, Viewer2D):
                    pos_list = self.viewer_mock.view.unscale_axis(*list(pos_list))
                    pos_list = list(pos_list)
                self.viewer_mock.set_crosshair_position(*pos_list)
        except Exception as e:
            pass
        return DataToActuators('outputs', mode='abs',
                               data=[DataActuator(self.modules_manager.actuators_name[ind],
                                                  data=float(outputs[ind])) for ind in
                                     range(len(outputs))])
