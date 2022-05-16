"""
Script to use jug to run through the different tasks in parallel.

See https://jug.readthedocs.io/en/latest/
You can also run them serially using the `run_hillclimbing.py` script instead.
"""

from jug import TaskGenerator

from syn_dags.script_utils import opt_utils
import run_hillclimbing

weight_path = '<fill this in !>'

@TaskGenerator
def run_ft(task):
    params = run_hillclimbing.Params(task, weight_path)
    res = run_hillclimbing.main(params)
    return res


task_list = [
    "guac_Perindopril_MPO"
]

out = [run_ft(task) for task in task_list]
