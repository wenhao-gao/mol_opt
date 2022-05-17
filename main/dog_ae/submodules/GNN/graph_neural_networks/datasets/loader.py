

from os import path
from pathlib import Path
import configparser

def get_qm9_data_path():
    home = str(Path.home())

    config_file = path.join(home, '.host_experiment_settings.ini')
    config = configparser.ConfigParser()
    config.read(config_file)

    return config['GNN']['qm9_data_path']