
import numpy as np
import torch

_settings_manager = None


class SettingsManager(object):
    def __init__(self):
        self.np_float_type = np.float32
        self.torch_float_type = torch.float32


def get_settings_manager() -> SettingsManager:
    global _settings_manager
    if _settings_manager is None:
        _settings_manager = SettingsManager()
    return _settings_manager
