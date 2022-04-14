from typing import Tuple
import scipy.stats
import numpy as np
from emukit.core.interfaces import IModel
from emukit.core.acquisition import Acquisition


class Max_GP(Acquisition):

    def __init__(self, model: IModel) -> None:
        """
        This acquisition computes for a given input point the value predicted by the GP surrogate model
        :param model: The underlying model that provides the predictive mean and variance for the given test points
        """
        self.model = model

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the predicted means
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        """
        mean, _ = self.model.predict(x)
        return -mean

    def evaluate_with_gradients(self, x: np.ndarray) -> Tuple:
        """
        Computes the predicted means and its derivative
        :param x: points where the acquisition is evaluated, shape (number of points, number of dimensions).
        """
        mean, variance = self.model.predict(x)

        dmean_dx, _ = self.model.get_prediction_gradients(x)
        return mean, dmean_dx

    @property
    def has_gradients(self):
        return isinstance(self.model, IDifferentiable)