
import numpy as np
import pymodaq.utils.math_utils as mutils


class RandomWrapper:

    Nx = 256
    amp = 20
    x0 = 128
    dx = 20
    n = 2

    def __init__(self):
        super().__init__()
        self._data = None
        self._current_value = 0.

    def get_value(self):
        return self._current_value

    def set_value(self, value: float = 0.):
        self._current_value = value

    def generate_data(self, x):
        data_mock = self.amp * (
            mutils.gauss1D(x, self.x0, self.dx,
                           self.n))
        data_mock += np.random.randn(*data_mock.shape) * self.amp / 3
        return data_mock

    def get_data_0D(self):
        return self.generate_data(self._current_value)

    def get_data_1D(self):
        x_axis = np.random.randint(0, self.Nx-1, self.Nx)
        return x_axis, self.generate_data(x_axis)
