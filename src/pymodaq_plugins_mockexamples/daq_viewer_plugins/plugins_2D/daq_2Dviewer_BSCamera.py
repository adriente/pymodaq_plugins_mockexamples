from qtpy.QtCore import QThread, Slot, QRectF
from qtpy import QtWidgets
import numpy as np
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, main, comon_parameters

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.parameter.utils import iter_children


from pymodaq_plugins_mockexamples.hardware.beam_steering import BeamSteering


class DAQ_2DViewer_BSCamera(DAQ_Viewer_base):

    live_mode_available = True
    hardware_averaging = True

    params = comon_parameters + [
        {'title': 'Wait time (ms)', 'name': 'wait_time', 'type': 'int', 'value': 100, 'min': 0},
    ]

    def ini_attributes(self):
        self.controller: BeamSteering = None
        self.live = False

    def commit_settings(self, param: Parameter):
        """
        """
        pass

    def ini_detector(self, controller=None):
        self.ini_detector_init(controller, BeamSteering())
        self.emit_status(ThreadCommand('update_main_settings',
                                       [['wait_time'], self.settings['wait_time'], 'value']))

        self.x_axis = Axis(data=self.controller.camera.x_axis, label='pixel', index=1)
        self.y_axis = Axis(data=self.controller.camera.y_axis, label='pixel', index=0)

        initialized = True
        info = 'Controller ok'
        return info, initialized

    def close(self):
        pass

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
        if 'live' in kwargs:
            if kwargs['live']:
                self.live = True
                # self.live = False  # don't want to use that for the moment

        if self.live:
            while self.live:
                data = self.average_data(Naverage)  # hardware averaging
                QThread.msleep(kwargs.get('wait_time', 100))
                self.dte_signal.emit(data)
                QtWidgets.QApplication.processEvents()
        else:
            data = self.average_data(Naverage)  # hardware averaging
            QThread.msleep(000)
            self.dte_signal.emit(data)

    def average_data(self, Naverage, init=False):
        data_tmp = np.zeros_like(self.controller.get_camera_data())
        for ind in range(Naverage):
            data_tmp += self.controller.get_camera_data()
        data_tmp = data_tmp / Naverage

        dwa = DataFromPlugins(name='BSCamera', data=[data_tmp],
                              axes=[self.x_axis, self.y_axis])

        return DataToExport('BSCamera', data=[dwa])

    def stop(self):
        """
            not implemented.
        """
        self.live = False
        return ""


if __name__ == '__main__':
    main(__file__)
