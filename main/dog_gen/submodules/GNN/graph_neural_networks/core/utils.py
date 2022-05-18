
import pickle

def find_inverse_dict(dict_in):
    return {value:key for key, value in dict_in.items()}


def to_pickle(obj_to_dump, filename):
    with open(filename, 'wb') as fo:
        pickle.dump(obj_to_dump, fo)


def from_pickle(filename):
    with open(filename, 'rb') as fo:
        data = pickle.load(fo)
    return data


class AverageMeter(object):
    """Computes and stores the average and current value

    taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L95-L113"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
