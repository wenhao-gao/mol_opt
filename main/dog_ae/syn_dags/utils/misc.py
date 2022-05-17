
import collections
import time
import pickle
import json


def try_but_pass(fn, exception, print_flag: bool=True):
    try:
        out = fn()
    except Exception as ex:
        if print_flag:
            print(ex)
        out = None
    return out




def retry_n_times(fn, n, exception=Exception, interval=0, on_exception=None, args=(), kwargs=None):
    """

    """
    if kwargs is None:
        kwargs = {}  # dictionaries are mutable so not ideal for default parameter values.

    for i in range(n):
        if i == n-1:  # Last try so don't catch exception
            return fn(*args, **kwargs)
        try:
            return fn(*args, **kwargs)
        except exception as e:
            if interval > 0:
                time.sleep(interval)
            if on_exception:
                on_exception(e)


def unpack_class_into_params_dict(params_in, prepender=""):
    if not isinstance(params_in, dict):
        params_in = vars(params_in)

    out_dict = {}
    def unpack(d_in, prepender_=""):
        for key, value in d_in.items():
            if isinstance(value, (int, float, str)):
                out_dict[prepender_ + str(key)] = value
            elif isinstance(value, dict):
                unpack(value, prepender_=f"{prepender_ + str(key)}:")
            elif isinstance(value, list):
                out_dict[prepender_ + str(key)] = str(value)
            else:
                pass
    unpack(params_in, prepender)
    return out_dict


def to_pickle(data, filepath):
    with open(filepath, 'wb') as fo:
        pickle.dump(data, fo)


def from_pickle(filepath):
    with open(filepath, 'rb') as fo:
        d = pickle.load(fo)
    return d


def load_json(path):
    with open(path, 'r') as fo:
        d = json.load(fo)
    return d
