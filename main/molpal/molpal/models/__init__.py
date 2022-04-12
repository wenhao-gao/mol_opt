"""This module contains the Model ABC and various implementations thereof. A
model is used to predict an input's objective function based on prior
training data."""

from typing import Optional, Set, Type

from main.molpal.molpal.models.base import Model

def model(model: str, **kwargs) -> Type[Model]:
    """Model factory function"""
    if model == 'rf':
        from molpal.models.sklmodels import RFModel

        return RFModel(**kwargs)

    if model == 'gp':
        from molpal.models.sklmodels import GPModel
        return GPModel(**kwargs)
        
    if model == 'nn':
        return nn(**kwargs)

    if model == 'mpn':
        return mpn(**kwargs)
    
    if model == 'random':
        from molpal.models.random import RandomModel
        return RandomModel(**kwargs)

    raise NotImplementedError(f'Unrecognized model: "{model}"')

def nn(conf_method: Optional[str] = None, **kwargs) -> Type[Model]:
    """NN-type Model factory function"""
    from molpal.models.nnmodels import (
        NNModel, NNDropoutModel, NNEnsembleModel, NNTwoOutputModel
    )

    try:
        return {
            'dropout': NNDropoutModel,
            'ensemble': NNEnsembleModel,
            'twooutput': NNTwoOutputModel,
            'mve': NNTwoOutputModel,
            'none': NNModel
        }.get(conf_method, 'none')(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f'Unrecognized NN confidence method: "{conf_method}"')

def mpn(conf_method: Optional[str] = None, **kwargs) -> Type[Model]:
    """MPN-type Model factory function"""
    from molpal.models.mpnmodels import (
        MPNModel, MPNDropoutModel, MPNTwoOutputModel
    )
            
    try:
        return {
            'dropout': MPNDropoutModel,
            'twooutput': MPNTwoOutputModel,
            'mve': MPNTwoOutputModel,
            'none': MPNModel
        }.get(conf_method, 'none')(conf_method=conf_method, **kwargs)
    except KeyError:
        raise NotImplementedError(
            f'Unrecognized MPN confidence method: "{conf_method}"')

def model_types() -> Set[str]:
    return {'rf', 'gp', 'nn', 'mpn'}
