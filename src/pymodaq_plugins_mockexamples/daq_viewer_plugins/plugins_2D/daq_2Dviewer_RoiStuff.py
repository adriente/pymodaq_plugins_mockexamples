from qtpy.QtCore import QThread, Slot, QRectF
from qtpy import QtWidgets
import numpy as np
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, main, comon_parameters

from pymodaq.utils.plotting.utils.plot_utils import RoiInfo
from pymodaq.utils.data import DataToExport, DataWithAxes
from pymodaq_plugins_mockexamples.daq_viewer_plugins.plugins_2D.daq_2Dviewer_MockCamera import DAQ_2DViewer_MockCamera


class DAQ_2DViewer_RoiStuff(DAQ_2DViewer_MockCamera):

    params = DAQ_2DViewer_MockCamera.params + \
             [{'title': 'USe Roi:', 'name': 'use_roi', 'type': 'bool'}]

    def ini_attributes(self):
        super().ini_attributes()
        self.roi_select_info: RoiInfo = None
        self.roi_select_viewer_index: int = None

    def ROISelect(self, info: QRectF):
        raise DeprecationWarning('DO not use it anymore, use the roi_select method')

    def roi_select(self, roi_info: RoiInfo, ind_viewer: int = 0):
        self.roi_select_info = roi_info
        self.roi_select_viewer_index = ind_viewer

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
                data: DataToExport = self.average_data(Naverage)  # hardware averaging
                QThread.msleep(kwargs.get('wait_time', 100))
                if self.settings['use_roi']:
                    dte = DataToExport('cropped')
                    # for dwa in data:
                    #     # dwa.
                    #     dte.append(dwa.isig[])
                self.dte_signal.emit(data)
                QtWidgets.QApplication.processEvents()
        else:
            data = self.average_data(Naverage)  # hardware averaging
            QThread.msleep(000)

        self.live = False  # don't want to use that for the moment

        data: DataToExport = self.average_data(Naverage)  # hardware averaging
        QThread.msleep(kwargs.get('wait_time', 100))
        if self.settings['use_roi']:
            self.roi_select_info.center_origin()
            dte = DataToExport('cropped')
            for dwa in data.data:
                dte.data.append(dwa.isig[self.roi_select_info.to_slices()])
        else:
            dte = data
        self.dte_signal.emit(dte)


if __name__ == '__main__':
    main(__file__)
