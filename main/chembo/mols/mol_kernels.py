"""
Molecular kernels.
To be used as part of CartesianProductKernel

Kernels to be implemented:
* Graph-based
* Fingerprints as vectors
* Fingerprints for molecular similarity
* String-based

TODO:
* Implement the remaining graph-based kernels
* Graphlets do not work
* For fingerprints, do projection

"""

import numpy as np
from typing import List, Union
import re
import logging
try:
    import graphkernels.kernels as gk
except ImportError as e:
    gk = None
    logging.info("Graphkernels package is not available, don't use graph-based kernels.")
from dragonfly.gp.kernel import Kernel, MaternKernel, ExpSumOfDistsKernel, SumOfExpSumOfDistsKernel
from myrdkit import DataStructs
from mols.molecule import Molecule


MOL_GRAPH_CONT_KERNEL_TYPES = [
    "edgehist_kernel", "vertexhist_kernel", "vehist_kernel"
]
MOL_GRAPH_INT_KERNEL_TYPES = [
    "vvehist_kernel", "edgehistgauss_kerenl",
    "vertexhistgauss_kernel", "vehistgauss_kernel", "georandwalk_kernel", "exprandwalk_kernel",
    "steprandwalk_kernel", "wl_kernel", "graphlet_kernel", "conngraphlet_kernel", "shortestpath_kernel"
]
MOL_FINGERPRINT_KERNEL_TYPES = ["fingerprint_kernel"]
MOL_SIMILARITY_KERNEL_TYPES = ["similarity_kernel"]
MOL_DISTANCE_KERNEL_TYPES = ["distance_kernel_expsum", "distance_kernel_sumexpsum", "distance_kernel_matern"]
MOL_SUM_KERNEL_TYPES = ["sum_kernel"]


def mol_kern_factory(kernel_type: str, *args, **kwargs):
    """
    factory method for generate a proper kernel
    :param kernel_type:
    :return: a proper kernel with `args` and `kwargs` that matches `kernel_type`
    """
    kernel_to_kernel_type = {
        MolGraphKernel: MOL_GRAPH_CONT_KERNEL_TYPES + MOL_GRAPH_INT_KERNEL_TYPES,
        MolFingerprintKernel: MOL_FINGERPRINT_KERNEL_TYPES,
        MolDistanceKernel: MOL_DISTANCE_KERNEL_TYPES,
        MolSimilarityKernel: MOL_SIMILARITY_KERNEL_TYPES,
        MolSumKernel: MOL_SUM_KERNEL_TYPES
    }
    kernel_type_to_kernel = {
        kernel_type: kernel
        for kernel, kernel_type_list in kernel_to_kernel_type.items()
        for kernel_type in kernel_type_list
    }
    if kernel_type not in kernel_type_to_kernel:
        raise ValueError("Not recognized kernel type: {}".format(kernel_type))
    kernel = kernel_type_to_kernel[kernel_type]
    return kernel(kernel_type, *args, **kwargs)


class MolKernel(Kernel):
    def __init__(self, kernel_type: str, **kwargs):
        self.kernel_type = kernel_type
        super(MolKernel, self).__init__()

    def is_guaranteed_psd(self):
        return False


class MolGraphKernel(MolKernel):
    if gk is not None:
        _kernel_calculator = {
            "edgehist_kernel": gk.CalculateEdgeHistKernel,
            "vertexhist_kernel": gk.CalculateVertexHistKernel,
            "vehist_kernel": gk.CalculateVertexEdgeHistKernel,
            "vvehist_kernel": gk.CalculateVertexVertexEdgeHistKernel,
            "vertexhistgauss_kernel": gk.CalculateVertexHistGaussKernel,
            "vehistgauss_kernel": gk.CalculateVertexEdgeHistGaussKernel,
            "georandwalk_kernel": gk.CalculateGeometricRandomWalkKernel,
            "exprandwalk_kernel": gk.CalculateExponentialRandomWalkKernel,
            "steprandwalk_kernel": gk.CalculateKStepRandomWalkKernel,
            "wl_kernel": gk.CalculateWLKernel,
            "graphlet_kernel": gk.CalculateGraphletKernel,
            "conngraphlet_kernel": gk.CalculateConnectedGraphletKernel,
            "shorestpath_kernel": gk.CalculateShortestPathKernel
        }

    def __init__(self, kernel_type: str, par: Union[int, float], **kwargs):
        """
        :param kernel_type: graph kernel type, refer to "https://github.com/BorgwardtLab/GraphKernels"
        :param par: `int` for integer parametrized graph kernels
                    `float` for float parametrized graph kernels
        """
        print(gk, gk==None)
        self._kernel_calculator = {
                "edgehist_kernel": gk.CalculateEdgeHistKernel,
                "vertexhist_kernel": gk.CalculateVertexHistKernel,
                "vehist_kernel": gk.CalculateVertexEdgeHistKernel,
                "vvehist_kernel": gk.CalculateVertexVertexEdgeHistKernel,
                "vertexhistgauss_kernel": gk.CalculateVertexHistGaussKernel,
                "vehistgauss_kernel": gk.CalculateVertexEdgeHistGaussKernel,
                "georandwalk_kernel": gk.CalculateGeometricRandomWalkKernel,
                "exprandwalk_kernel": gk.CalculateExponentialRandomWalkKernel,
                "steprandwalk_kernel": gk.CalculateKStepRandomWalkKernel,
                "wl_kernel": gk.CalculateWLKernel,
                "graphlet_kernel": gk.CalculateGraphletKernel,
                "conngraphlet_kernel": gk.CalculateConnectedGraphletKernel,
                "shorestpath_kernel": gk.CalculateShortestPathKernel
            }


        super(MolGraphKernel, self).__init__(kernel_type, **kwargs)
        self.set_hyperparams(par=par)
        if kernel_type not in self._kernel_calculator:
            raise ValueError("Unknown kernel_type {}".format(kernel_type))
        self.kernel_calculator = self._kernel_calculator[kernel_type]

    def _child_evaluate(self, X1: List[Molecule], X2: List[Molecule]) -> np.array:
        complete_graph_list = [m.to_graph() for m in X1 + X2]
        if self.kernel_type in MOL_GRAPH_INT_KERNEL_TYPES:
            par = int(self.hyperparams["par"])
        else:
            par = self.hyperparams["par"]
        complete_ker = self.kernel_calculator(complete_graph_list, par=par)
        n1 = len(X1)
        return complete_ker[:n1, n1:]


class MolSimilarityKernel(MolKernel):
    """ Kernel using fingerprint representations 
        and pre-defined similarity of them.
    """
    def __init__(self, kernel_type: str, **kwargs):
        """
        :param kernel_type: graph kernel type, currently just one
                            ("similarity_kernel")
        """
        super(MolSimilarityKernel, self).__init__(kernel_type, **kwargs)

    def _get_fps(self, X):
        """
        turn each molecule to its fingerprint representation
        """
        return [mol.to_fingerprint(ftype="fp") for mol in X]

    def _construct_sim_matrix(self, fps):
        """ X - list of fps """
        rows = []
        nfps = len(fps)
        for i in range(nfps):
            sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
            rows.append(sims)
        res = np.array(rows)
        assert res.shape == (nfps, nfps)
        return res

    def _child_evaluate(self, X1, X2):
        X1 = self._get_fps(X1)
        X2 = self._get_fps(X2)
        complete_ker = self._construct_sim_matrix(X1 + X2)
        n1 = len(X1)
        return complete_ker[:n1, n1:]


class MolDistanceKernel(MolKernel):
    """
    evaluate kernel based on the distance measure between molecules
    """
    def __init__(self, kernel_type: str, **kwargs):
        super(MolDistanceKernel, self).__init__(kernel_type, **kwargs)
        base_kernel_type = self.get_base_kernel_type(kernel_type)
        if base_kernel_type == "expsum":
            self.base_kernel = ExpSumOfDistsKernel(
                dist_computer=kwargs["dist_computer"],
                betas=kwargs["betas"],
                scale=1.0  # scale is 1.0 as only the product kernel has scale
            )
        elif base_kernel_type == "sumexpsum":
            raise NotImplementedError
        elif base_kernel_type == "matern":
            raise NotImplementedError
        else:
            raise ValueError("{} not implemented for distance kernel".format(base_kernel_type))

    @staticmethod
    def get_base_kernel_type(kernel_type):
        """
        :param kernel_type: as of form `"distance_kernel_{base_kernel_type}"`
        :return: `base_kernel_type`
        """
        p = re.compile("distance_kernel_(.*)")
        base_kernel_type = p.match(kernel_type).group(1)
        return base_kernel_type

    def is_guaranteed_psd(self):
        return self.base_kernel.is_guaranteed_psd()

    def evaluate_from_dists(self, dists: List[np.array]):
        return self.base_kernel.evaluate_from_dists(dists)

    def _child_evaluate(self, X1, X2):
        return self.base_kernel.evaluate(X1, X2)

    def __str__(self):
        return "MolDistanceKernel: " + str(self.base_kernel)


class MolFingerprintKernel(MolKernel):
    """ Kernel based on vectorized representations of molecules.
        In TODO mode.
    """
    def __init__(self, kernel_type: str, base_kernel: Kernel, **kwargs):
        super(MolFingerprintKernel, self).__init__(kernel_type, **kwargs)
        self.base_kernel = base_kernel

    def is_guaranteed_psd(self):
        return self.base_kernel.is_guaranteed_psd()

    def _get_fps(self, X: List[Molecule]):
        res = np.array([mol.to_fingerprint() for mol in X])
        return res

    def _child_evaluate(self, X1: List[Molecule], X2: List[Molecule]):
        X1 = self._get_fps(X1)
        X2 = self._get_fps(X2)
        return self.base_kernel.evalute(X1, X2)

    def __str__(self):
        return "FingerprintKernel: " + str(self.base_kernel)


class MolSumKernel(MolKernel):
    """
    Molecule kernel that is a weighted sum of scaled kernels.
    k(x,y) = alpha1 * k1(x,y) + ... alphan * kn(x,y)
    """
    # TODO: careful about the order of arguments here
    def __init__(self, kernel_type, alphas, dist_computer, betas, **kwargs):
        """ TODO: this constructor could be made more general """
        super(MolSumKernel, self).__init__(kernel_type, **kwargs)
        self.kernels = [
            MolSimilarityKernel("similarity_kernel"),
            MolDistanceKernel("distance_kernel_expsum",
                dist_computer=dist_computer, betas=betas)
            ]
        self.add_hyperparams(alphas=np.array(alphas), betas=np.array(betas))

        # Array to keep largest kernel values
        # NOT USED YET
        self.max_values_seen = [-float('inf')] * len(self.kernels)

    def is_guaranteed_psd(self):
        return all(kernel.is_guaranteed_psd() for kernel in self.kernels)

    def evaluate_from_dists(self, dists: List[np.array], X1, X2):
        """ TODO: this method could be made more general """
        # kernel matrix from the distance-based kernel
        sim_kernel_mat = self.kernels[0](X1, X2)
        # kernel matrix from the distance-based kernel
        dist_kernel_mat = self.kernels[1].evaluate_from_dists(dists)
        alpha0, alpha1 = self.hyperparams['alphas']
        # Log the hyperparameter optimization only some fraction of times:
        if np.random.binomial(1, 0.1):
            logging.debug(f"Alphas: {self.hyperparams['alphas']}, log betas: {self.hyperparams['betas']}")
        return alpha0 * sim_kernel_mat + alpha1 * dist_kernel_mat

    def _child_evaluate(self, X1, X2):
        """ Shouldn't be used as of now (and in general,
        in the cases when there is a distance-based component)"""
        sum_kernel_mat = 0.
        for kernel, alpha in zip(self.kernels, self.hyperparams['alphas']):
            sum_kernel_mat += alpha * kernel._child_evaluate(X1, X2)
        return sum_kernel_mat

    def __str__(self):
        return "Sum kernel of " + ",".join([str(kernel) for kernel in self.kernels])


class MolStringKernel(MolKernel):
    # TODO: implement this
    pass

