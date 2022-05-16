
import functools

from torch.utils.tensorboard import SummaryWriter

_tb_writer = {}


def get_tb_writer(path):
    global _tb_writer

    if path not in _tb_writer:
        _tb_writer[path] = SummaryWriter(path)

        # We will monkey patch the Summary Writer so that it can use the (new) step attribute we added to the class
        # if it does not yet exist.

        def create_new_func(wrapped, instance):
            @functools.wraps(wrapped)
            def add_event(*args, **kwargs):
                kwargs.update(zip(wrapped.__code__.co_varnames[1:], args))
                step = kwargs.get("step", None)
                if step is None:
                    try:
                        step = _tb_writer[path].global_step
                    except AttributeError:
                        pass
                kwargs['step'] = step
                return wrapped(**kwargs)
            return add_event

        _tb_writer[path].file_writer.add_event = create_new_func(_tb_writer[path].file_writer.add_event,
                                                                 _tb_writer[path].file_writer)

    return _tb_writer[path]