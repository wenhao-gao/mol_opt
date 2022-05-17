
import pickle
import enum

from torch.utils import data



class DatasetPartitions(enum.Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"

