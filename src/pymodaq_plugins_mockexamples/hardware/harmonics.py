import numpy as np

from pymodaq_utils import math_utils as mutils
from pymodaq_utils.units import nm2eV, eV2nm

from pymodaq_data import Q_
from pymodaq_plugins_mock.hardware.wrapper import ActuatorWrapperWithTau


class Harmonics(ActuatorWrapperWithTau):

    def __init__(self):
        super().__init__()
        self._n_harmonics = 3
        self._omega0 = Q_(nm2eV(800), 'eV')
        self._domega_ev = Q_(0.2, 'eV')
        self._npts = 512
        self._current_value = 1.0
        self._target_value = 1.0

    @property
    def amplitude(self) -> float:
        return self.get_value()

    @amplitude.setter
    def amplitude(self, amp: float):
        self.move_at(amp)

    @property
    def n_harmonics(self) -> int:
        return self._n_harmonics

    @n_harmonics.setter
    def n_harmonics(self, nhar: int):
        if isinstance(nhar, int):
            self._n_harmonics = nhar

    def get_axis(self) -> Q_:
        return np.linspace(0, (self._n_harmonics + 1) * self._omega0, self._npts)

    def get_spectrum(self) -> Q_:
        axis = self.get_axis()
        spectrum = Q_(np.zeros((self._npts,)))
        for ind in range(self._n_harmonics):
            spectrum += mutils.gauss1D(axis, (ind+1) * self._omega0, self._domega_ev)
        spectrum *= self._current_value
        spectrum += 0.1 * np.random.random_sample((self._npts,))
        return spectrum
