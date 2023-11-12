from pathlib import Path
import queue
import time
import tempfile
from threading import Lock

import numpy as np
from qtpy import QtCore

from pymodaq.utils.daq_utils import ThreadCommand
from pymodaq.utils.data import DataFromPlugins, Axis, DataToExport, DataRaw, DataCalculated
from pymodaq.control_modules.viewer_utility_classes import DAQ_Viewer_base, comon_parameters, main
from pymodaq.utils.parameter import Parameter
from pymodaq.utils.parameter.utils import iter_children
from pymodaq_plugins_mockexamples.hardware.photon_yielder import PhotonYielder, Photons
from pymodaq.utils.h5modules.saving import H5Saver
from pymodaq.utils.h5modules.data_saving import (DataEnlargeableSaver, DataToExportEnlargeableSaver, DataLoader,
                                                 NodeError)

lock = Lock()

class DAQ_NDViewer_MockEvents(DAQ_Viewer_base):
    """ Instrument plugin class for a 2D viewer.
    
    This object inherits all functionalities to communicate with PyMoDAQ’s DAQ_Viewer module through inheritance via
    DAQ_Viewer_base. It makes a bridge between the DAQ_Viewer module and the Python wrapper of a particular instrument.

    Attributes:
    -----------
    controller: PhotonYielder
        The particular object that allow the communication with the hardware, in general a python wrapper around the
         hardware library.
    """
    grabber_start_signal = QtCore.Signal(float, int)
    saver_start_signal = QtCore.Signal()

    params = comon_parameters + [
        {'title': 'Acqu. Time (s)', 'name': 'acqui_time', 'type': 'float', 'value': 10},
        {'title': 'Refresh Time (ms)', 'name': 'refresh_time', 'type': 'int', 'value': 500},
        {'title': 'Wait Time (ms)', 'name': 'wait_time', 'type': 'int', 'value': 1},
        {'title': 'Show ND', 'name': 'show_nd', 'type': 'bool_push', 'value': False, 'label': 'Show ND'},
        {'title': 'Histogram:', 'name': 'histogram', 'type': 'group', 'children': [
            {'title': 'Apply weight?', 'name': 'apply_weight', 'type': 'bool', 'value': False},
            {'title': 'Time min (µs)', 'name': 'time_min', 'type': 'float', 'value': 0},
            {'title': 'Time max (µs)', 'name': 'time_max', 'type': 'float', 'value': 20},
            {'title': 'N Bin Time', 'name': 'nbin_time', 'type': 'int', 'value': 256},
            {'title': 'N Bin X', 'name': 'nbin_x', 'type': 'int', 'value': 256},
            {'title': 'N Bin y', 'name': 'nbin_y', 'type': 'int', 'value': 256},
        ]},
    ]

    def ini_attributes(self):
        self.controller: PhotonYielder = None

        self.x_axis: Axis = None
        self.y_axis: Axis = None
        self.time_axis: Axis = None

        self.h5temp : H5Saver() = None
        self.temp_path : tempfile.TemporaryDirectory = None
        self.saver: DataToExportEnlargeableSaver = None
        self._loader: DataLoader = None

        self.saver_thread: QtCore.QThread = None

        self._queue = queue.Queue()

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_events)

    def commit_settings(self, param: Parameter):
        """Apply the consequences of a change of value in the detector settings

        Parameters
        ----------
        param: Parameter
            A given parameter (within detector_settings) whose value has been changed by the user
        """
        # TODO for your custom plugin
        if param.name() in iter_children(self.settings.child('histogram'), []):
            self.process_events()
        elif param.name() == 'show_nd':
            if param.value():
                self.show_nd()
                param.setValue(False)


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
                               new_controller=PhotonYielder())

        callback = PhotonCallback(self.controller, self._queue)
        self.callback_thread = QtCore.QThread()  # creation of a Qt5 thread
        callback.moveToThread(self.callback_thread)  # callback object will live within this thread
        callback.data_sig.connect(self.emit_data)

        self.grabber_start_signal.connect(callback.grab)
        self.callback_thread.callback = callback

        self.callback_thread.start()

        self.dte_signal_temp.emit(DataToExport('Events', data=[DataCalculated('TOF', data=[np.array([0, 1, 2])],
                                                                              )
                                                               ]
                                               ))

        info = "Whatever info you want to log"
        initialized = True
        return info, initialized

    def emit_data(self):
        self.timer.stop()
        print('acquisition is done')

    def process_events(self):
        try:
            node = self._loader.get_node('/RawData/myphotons/DataND/CH00/EnlData00')
            lock.acquire()
            dwa = self._loader.load_data(node, load_all=True)
            lock.release()
            print(f'Nphotons: {dwa.size}')
            dwa_tof = self.compute_histogram(dwa, '1D')
            self.dte_signal_temp.emit(DataToExport('TOF', data=[dwa_tof]))
        except NodeError:
            pass

    def compute_histogram(self, dwa: DataRaw, dim='1D'):
        if dim == '1D':
            time_of_flight, time_array = np.histogram(dwa.axes[0].get_data(),
                                                      bins=self.settings['histogram', 'nbin_time'],
                                                      range=(self.settings['histogram', 'time_min'] * 1e-6,
                                                             self.settings['histogram', 'time_max'] * 1e-6),
                                                      weights=dwa.data[0] if self.settings['histogram', 'apply_weight']
                                                      else None)

            dwa_tof = DataCalculated('TOF', data=[time_of_flight],
                                     axes=[Axis('Time', 's', time_array[:-1])])

        else:
            data_array, edges = np.histogramdd(np.stack((dwa.axes[0].get_data(),
                                                         np.squeeze(dwa[1].astype(int)),
                                                         np.squeeze(dwa[2]).astype(int)), axis=1),
                                               bins=(self.settings['histogram', 'nbin_time'],
                                                     self.settings['histogram', 'nbin_x'],
                                                     self.settings['histogram', 'nbin_y']),
                                               range=((self.settings['histogram', 'time_min'] * 1e-6,
                                                      self.settings['histogram', 'time_max'] * 1e-6),
                                                      None,
                                                      None)
                                               )
            dwa_tof = DataCalculated('TOF', data=[data_array],
                                     axes=[Axis('Time', 's', edges[0][:-1], index=0),
                                           Axis('X', 's', edges[1][:-1], index=1),
                                           Axis('Y', 's', edges[2][:-1], index=2)],
                                     nav_indexes=(1, 2))
        return dwa_tof

    def show_nd(self):
        node = self._loader.get_node('/RawData/myphotons/DataND/CH00/EnlData00')
        dwa = self._loader.load_data(node, load_all=True)

        dwa_tof = self.compute_histogram(dwa, 'ND')
        self.dte_signal_temp.emit(DataToExport('TOF', data=[dwa_tof]))

    def close(self):
        """Terminate the communication protocol"""
        self.saver.close()

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
        if self.h5temp is not None:
            self.h5temp.close()
            self.temp_path.cleanup()
        if self.saver_thread is not None:
            if self.saver_thread.isRunning():
                self.saver_thread.terminate()
                while self.saver_thread.isRunning():
                    QtCore.QThread.msleep(100)
                    print('Thread still running')

        self.h5temp = H5Saver(save_type='detector')
        self.temp_path = tempfile.TemporaryDirectory(prefix='pymo')
        addhoc_file_path = Path(self.temp_path.name).joinpath('temp_data.h5')
        self.h5temp.init_file(custom_naming=True, addhoc_file_path=addhoc_file_path)
        self.h5temp.get_set_group('/RawData', 'myphotons')
        self.saver: DataToExportEnlargeableSaver = DataToExportEnlargeableSaver(self.h5temp,
                                                                                axis_name='photon index',
                                                                                axis_units='index')
        self._loader = DataLoader(self.h5temp)

        self.controller.ind_grabed = -1

        save_callback = SaverCallback(self._queue, self.saver)
        self.saver_thread = QtCore.QThread()
        save_callback.moveToThread(self.saver_thread)
        self.saver_thread.callback = save_callback
        self.saver_start_signal.connect(save_callback.work)
        self.saver_thread.start()

        self.grabber_start_signal.emit(self.settings['acqui_time'], self.settings['wait_time'])
        self.saver_start_signal.emit()
        self.timer.setInterval(self.settings['refresh_time'])
        self.timer.start()

    def stop(self):
        """Stop the current grab hardware wise if necessary"""
        return ''


class PhotonCallback(QtCore.QObject):
    """

    """
    data_sig = QtCore.Signal()

    def __init__(self, photon_grabber: PhotonYielder, event_queue: queue.Queue):
        super().__init__()
        self.photon_grabber = photon_grabber
        self.event_queue = event_queue

    def grab(self, acquisition_time: float, wait_time: int):
        start_acqui = time.perf_counter()
        while (time.perf_counter() - start_acqui) <= acquisition_time:
            photons: Photons = self.photon_grabber.grab()
            self.event_queue.put(photons)
            QtCore.QThread.msleep(wait_time)
        self.data_sig.emit()


class SaverCallback(QtCore.QObject):
    def __init__(self, event_queue: queue.Queue, saver: DataToExportEnlargeableSaver):
        super().__init__()
        self.event_queue = event_queue
        self.saver = saver

    def work(self):
        while True:
            photons: Photons = self.event_queue.get()
            data = DataToExport('photons', data=[
                DataRaw('time', data=[photons.intensity, photons.x_pos, photons.y_pos],
                        labels=['intensity', 'x_pos', 'y_pos'],
                        nav_indexes=(0, ),
                        axes=[Axis('timestamps', data=photons.time_stamp, index=0)]
                        )
            ])
            lock.acquire()
            self.saver.add_data('/RawData/myphotons', axis_value=photons.time_stamp, data=data)
            lock.release()
            self.event_queue.task_done()


if __name__ == '__main__':
    main(__file__)
