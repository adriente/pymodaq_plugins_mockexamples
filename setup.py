from setuptools import setup, find_packages
import toml

config = toml.load('./plugin_info.toml')
SHORT_PLUGIN_NAME = config['plugin-info']['SHORT_PLUGIN_NAME']
PLUGIN_NAME = f"pymodaq_plugins_{SHORT_PLUGIN_NAME}"

from pathlib import Path

setup(Path(__file__).parent)
