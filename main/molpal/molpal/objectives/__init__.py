from typing import Type

from main.molpal.molpal.objectives.base import Objective

def objective(objective, objective_config: str, **kwargs) -> Type[Objective]:
    """Objective factory function"""
    if objective == 'docking':
        from molpal.objectives.docking import DockingObjective
        return DockingObjective(objective_config, **kwargs)
    if objective == 'lookup':
        from molpal.objectives.lookup import LookupObjective
        return LookupObjective(objective_config, **kwargs)
    
    raise NotImplementedError(f'Unrecognized objective: "{objective}"')
