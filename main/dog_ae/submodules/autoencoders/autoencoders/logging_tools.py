
import typing


class LogHelper:
    def __init__(self, funcs_to_call: typing.List[typing.Callable[[typing.Mapping[str, typing.Any]], None]]):
        self.funcs_to_call = funcs_to_call

    def should_run_collect_extra_statistics(self):
        return bool(len(self.funcs_to_call))

    def add_statistics(self, statistics: dict):
        """
        keys of dict should indicate what kind of value it is:
        eg
        * "raw-<expression without hyphens>" raw scalar.
        * "sum-<expression without hyphens>" summed (over batch) scalar.
        """
        for func in self.funcs_to_call:
            func(statistics)
