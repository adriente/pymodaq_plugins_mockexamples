from pathlib import Path
from pymodaq_utils.logger import set_logger, get_module_name
from pymodaq_utils.resources.hatch_build_plugins import PluginInfoTomlHook

here = Path(__file__).absolute().parent
logger = set_logger(get_module_name(__file__))


class PluginInfoTomlHook(PluginInfoTomlHook):
    def update(self, metadata: dict) -> None:
        super().update_custom(metadata, here)
