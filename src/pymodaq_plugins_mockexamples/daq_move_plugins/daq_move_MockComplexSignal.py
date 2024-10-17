from pymodaq.control_modules.move_utility_classes import DAQ_Move_base, main  # base class
from pymodaq.control_modules.move_utility_classes import comon_parameters_fun  # common set of parameters for all actuators

from pymodaq.utils.daq_utils import ThreadCommand, getLineInfo  # object used to send info back to the main thread
from pymodaq_plugins_mockexamples.hardware.complex_signal import DataSignal

from pymodaq_plugins_mockexamples import config


class DAQ_Move_MockComplexSignal(DAQ_Move_base):
    """
        Wrapper object to access the Mock fonctionnalities, similar wrapper for all controllers.

        =============== ==============
        **Attributes**    **Type**
        *params*          dictionnary
        =============== ==============
    """
    _controller_units = ''
    is_multiaxes = True
    _axis_names = dict(zip(DataSignal.axes, list(range(len(DataSignal.axes)))))
    _epsilon = 0.01

    params = [{'title': 'Tau (ms):', 'name': 'tau', 'type': 'int',
               'value': DataSignal._tau * 1000, 'tip': 'Characteristic evolution time'},] + \
             comon_parameters_fun(is_multiaxes, axes_names=_axis_names, epsilon=_epsilon)

    def ini_attributes(self):
        self.controller: DataSignal = None

    def get_actuator_value(self):
        pos = self.controller.get_value(self.axis_name)
        pos = self.get_position_with_scaling(pos)
        return pos

    def close(self):
        pass

    def commit_settings(self, param):
        if param.name() == 'tau':
            self.controller.tau = param.value() / 1000  # controller need a tau in seconds while the param tau is in ms
        elif param.name() == 'epsilon':
            self.controller.epsilon = param.value()

    def ini_stage(self, controller=None):
        """
            Initialize the controller and stages (axes) with given parameters.

            ============== ================================================ ==========================================================================================
            **Parameters**  **Type**                                         **Description**

            *controller*    instance of the specific controller object       If defined this hardware will use it and will not initialize its own controller instance
            ============== ================================================ ==========================================================================================

            Returns
            -------
            Easydict
                dictionnary containing keys:
                 * *info* : string displaying various info
                 * *controller*: instance of the controller object in order to control other axes without the need to init the same controller twice
                 * *stage*: instance of the stage (axis or whatever) object
                 * *initialized*: boolean indicating if initialization has been done correctly

            See Also
            --------
             daq_utils.ThreadCommand
        """
        self.ini_stage_init(controller, DataSignal())
        if self.settings['multiaxes', 'multi_status'] == "Master":
            self.controller.ini_random_structures()
        info = "Mock MultiAxis"
        initialized = True
        return info, initialized

    def move_abs(self, position):
        position = self.check_bound(position)  #if user checked bounds, the defined bounds are applied here
        self.target_value = position
        position = self.set_position_with_scaling(position)
        pos = self.controller.move_at(position, self.axis_name)

    def move_rel(self, position):
        position = self.check_bound(self.current_position + position) - self.current_position
        self.target_value = position + self.current_position
        position = self.set_position_with_scaling(self.target_value)

        pos = self.controller.move_at(position, self.axis_name)

    def move_home(self):
        """
          Send the update status thread command.
            See Also
            --------
            daq_utils.ThreadCommand
        """
        self.emit_status(ThreadCommand('Update_Status', ['Move Home not implemented']))

    def stop_motion(self):
        """
          Call the specific move_done function (depending on the hardware).

          See Also
          --------
          move_done
        """
        self.move_done()


if __name__ == '__main__':
    main(__file__)
