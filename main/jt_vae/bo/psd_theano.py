import logging

import numpy
import scipy.linalg as spla
import theano
from theano.gof import Op, Apply
from theano.tensor import as_tensor_variable
from theano.tensor.nlinalg import matrix_dot

logger = logging.getLogger(__name__)

__all__ = ['MatrixInversePSD', 'LogDetPSD']


def chol2inv(chol):
    return spla.cho_solve((chol, False), numpy.eye(chol.shape[0]))


class MatrixInversePSD(Op):
    r"""Computes the inverse of a matrix :math:`A`.
    Given a square matrix :math:`A`, ``matrix_inverse`` returns a square
    matrix :math:`A_{inv}` such that the dot product :math:`A \cdot A_{inv}`
    and :math:`A_{inv} \cdot A` equals the identity matrix :math:`I`.
    Notes
    -----
    When possible, the call to this op will be optimized to the call
    of ``solve``.
    """

    __props__ = ()

    def __init__(self):
        pass

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        return Apply(self, [x], [x.type()])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        z[0] = chol2inv(spla.cholesky(x, lower=False)).astype(x.dtype)

    def grad(self, inputs, g_outputs):
        r"""The gradient function should return
            .. math:: V\frac{\partial X^{-1}}{\partial X},
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: (X^{-1} \cdot V^{T} \cdot X^{-1})^T.
        """
        x, = inputs
        xi = self(x)
        gz, = g_outputs
        # TT.dot(gz.T,xi)
        return [-matrix_dot(xi, gz.T, xi).T]

    def R_op(self, inputs, eval_points):
        r"""The gradient function should return
            .. math:: \frac{\partial X^{-1}}{\partial X}V,
        where :math:`V` corresponds to ``g_outputs`` and :math:`X` to
        ``inputs``. Using the `matrix cookbook
        <http://www2.imm.dtu.dk/pubdb/views/publication_details.php?id=3274>`_,
        one can deduce that the relation corresponds to
            .. math:: X^{-1} \cdot V \cdot X^{-1}.
        """
        x, = inputs
        xi = self(x)
        ev, = eval_points
        if ev is None:
            return [None]
        return [-matrix_dot(xi, ev, xi)]

    def infer_shape(self, node, shapes):
        return shapes


matrix_inverse_psd = MatrixInversePSD()


class LogDetPSD(Op):
    """
    Matrix log determinant. Input should be a square matrix.
    """

    __props__ = ()

    def make_node(self, x):
        x = as_tensor_variable(x)
        assert x.ndim == 2
        o = theano.tensor.scalar(dtype=x.dtype)
        return Apply(self, [x], [o])

    def perform(self, node, inputs, outputs):
        (x,) = inputs
        (z,) = outputs
        try:
            z[0] = numpy.asarray(2 * numpy.sum(
                numpy.log(numpy.diag(spla.cholesky(x, lower=False)))),
                                 dtype=x.dtype)
        except Exception:
            print('Failed to compute log determinant', x)
            raise

    def grad(self, inputs, g_outputs):
        gz, = g_outputs
        x, = inputs
        return [gz * matrix_inverse_psd(x).T]

    def infer_shape(self, node, shapes):
        return [()]

    def __str__(self):
        return "LogDetPSD"


log_det_psd = LogDetPSD()


