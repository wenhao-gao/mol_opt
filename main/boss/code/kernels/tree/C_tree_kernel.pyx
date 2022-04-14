# cython: profile=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False

import nltk
import numpy as np
cimport numpy as np
from collections import defaultdict
from libcpp.string cimport string
cdef extern from "math.h":
    double sqrt(double x)


# This code is a port of https://github.com/beckdaniel/GPy/blob/tk_master_nograds/GPy/kern/_src/cy_tree.pyx
# Our main changes are:
# 1) Upgrade code to work for python3 and GPy 1.9.9
# 2) We focus on a single highly efficient implementation of the SSTK kernel of Moschitti (2006)
# 3) We fine-tune the cython implementation to provide another order of computational speed ups
# 4) We improve the documentation


class wrapper_raw_SubsetTreeKernel(raw_SubsetTreeKernel):
    # Dummy wrapper to allow the main class to be cast into C, whilst being accessible from python
    pass

cdef class Node(object):
    """
    A node object, containing a grammar production, an id and the children ids.
    These are the nodes stored in the node lists implementation of the SSTK
    (the "Fast Tree Kernel" of Moschitti (2006))
    :param production: String of the grammar production of the node (e.g production of node S in '(S (NP ns) (VP v))' is 'S NP VP')
                        This will be useful for creating ids to store node info in a dictionary
    :param node_id: Unique ID of the Node
    :param children_ids: List of unique IDs of the Node's children
    """

    cdef str production
    cdef int node_id
    cdef list children_ids
    
    
    def __init__(self, str production, int node_id, list children_ids):
        self.production = production
        self.node_id = node_id
        self.children_ids = children_ids

    def __repr__(self):
        return str((self.production, self.node_id, self.children_ids))

cdef class raw_SubsetTreeKernel(object):
    """
    The "Fast Tree Kernel" of Moschitti (2006), with two parameters.
    Following Beck (2015) to get gradients wrt kernel parameters
    :param _lambda: lambda weights the contribtuion of largers tree fragments
    :param _sigma: sigma controls sparsity
    :param normalization: Bool to control if we normalize. If comparing trees of different depth then should normalize.

    """
    cdef int normalize
    cdef dict _tree_cache
    
    def __init__(self, double _lambda=1., double _sigma=1., bint normalize=True):
        self._lambda = _lambda
        self._sigma = _sigma
        self._tree_cache = {}
        self.normalize = normalize

    cdef tuple _gen_node_list(self, str tree_repr):
        """
        Generates an ordered list of nodes from a tree.
        The list is used to generate the node pairs when
        calculating K.
        It also returns a nodes dict for fast node access.
        """
        tree = nltk.tree.Tree.fromstring(tree_repr)

        cdef list node_list = []
        self._get_node(tree, node_list)
        node_list.sort(key=lambda Node x: x.production)
        cdef Node node
        cdef dict node_dict
        node_dict = dict([(node.node_id, node) for node in node_list])
        return node_list, node_dict
    

    cdef int _get_node(self, tree, list node_list):
        """
        Recursive method for generating the node lists.
        """
        cdef str cprod
        cdef Node node
        cdef int node_id, ch_id
        cdef list prod_list, children
        
        if type(tree[0]) != str:
            prod_list = [tree.label()]
            children = []
            for ch in tree:
                ch_id = self._get_node(ch, node_list)
                #prod_list.append(ch.node)
                prod_list.append(ch.label())
                children.append(ch_id)
            node_id = len(node_list)
            cprod = ' '.join(prod_list)
            node = Node(cprod, node_id, children)
            node_list.append(node)
            return node_id
        else:
            cprod = ' '.join([tree.label(), tree[0]])
            node_id = len(node_list)
            node = Node(cprod, node_id, None)
            node_list.append(node)
            return node_id            


    cdef void _build_cache(self, np.ndarray X):
        """
        Caches the node lists, for each tree that it is not
        in the cache. If all trees in X are already cached, this
        method does nothing.

        These node lists can then be quickly accessed when calculating K, rather than having to 
        traverse trees each time we want to access a node. This provided substantial speed ups (x10)
        """
        cdef np.ndarray tree_repr
        cdef str t_repr
        cdef list node_list
        cdef dict node_dict
        
        for tree_repr in X:
            t_repr = tree_repr[0]
            if t_repr not in self._tree_cache:
                node_list, node_dict = self._gen_node_list(t_repr)
                self._tree_cache[t_repr] = (node_list, node_dict)

    cpdef Kdiag(self, np.ndarray X):
        """
        The method that calls the SSTK for each individual tree.
        """
        cdef np.ndarray[np.double_t, ndim=1] X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas
        self._build_cache(X)
        X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
        return X_diag_Ks
        
    cpdef tuple K(self, np.ndarray X, np.ndarray X2):
        """
        The method that calls the SSTK for each tree pair. Some shortcuts are used
        when X2 == None (when calculating the Gram matrix for X).
        """
        cdef np.ndarray[np.double_t, ndim=1] X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas,X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas
        cdef np.ndarray x1, x2
        cdef np.ndarray[np.double_t, ndim=2] Ks, dlambdas, dsigmas 
        cdef int i,j
        cdef bint symmetric
        cdef list nodes1, nodes2
        cdef dict dict1, dict2
        cdef double K_result, dlambda, dsigma,K_norm, dlambda_norm, dsigma_norm

        # Put any new trees in the cache. If the trees are already cached, this code
        # won't do anything.
        self._build_cache(X)
        if X2 is None:
            symmetric = True
            X2 = X
        else:
            symmetric = False
            self._build_cache(X2)
            
            
        # Calculate K for diagonal values
        # because we will need them later to normalize.
        if self.normalize:
            X_diag_Ks, X_diag_dlambdas, X_diag_dsigmas = self._diag_calculations(X)
            if not symmetric:
                X2_diag_Ks, X2_diag_dlambdas, X2_diag_dsigmas = self._diag_calculations(X2)
            
        # Initialize the derivatives here 
        # because we are going to calculate them at the same time as K.

        Ks = np.zeros(shape=(len(X), len(X2)))
        dlambdas = np.zeros(shape=(len(X), len(X2)))
        dsigmas = np.zeros(shape=(len(X), len(X2)))

        # Iterate over the trees in X and X2 (or X and X in the symmetric case).
        for i, x1 in enumerate(X):
            for j, x2 in enumerate(X2):
                # Shortcut: no calculation is needed for the upper
                # part of the Gram matrix because it is symmetric
                if symmetric:
                    if i > j:
                        Ks[i][j] = Ks[j][i]
                        dlambdas[i][j] = dlambdas[j][i]
                        dsigmas[i][j] = dsigmas[j][i]
                        continue
                    # Another shortcut: because this is the normalized SSTK
                    # diagonal values will always be equal to 1.
                    if i == j and self.normalize:
                        Ks[i][j] = 1
                        continue
                
                # It will always be a 1-element array so we just index by 0
                nodes1, dict1 = self._tree_cache[x1[0]]
                nodes2, dict2 = self._tree_cache[x2[0]]
                K_result, dlambda, dsigma = self._calc_K(nodes1, nodes2, dict1, dict2)

                # Normalization happens here.
                if self.normalize:
                    if symmetric:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X_diag_dsigmas[j])
                    else:
                        K_norm, dlambda_norm, dsigma_norm = self._normalize(K_result, dlambda, dsigma,
                                                                            X_diag_Ks[i], X2_diag_Ks[j],
                                                                            X_diag_dlambdas[i], X2_diag_dlambdas[j],
                                                                            X_diag_dsigmas[i], X2_diag_dsigmas[j])

                # Store everything, including derivatives.
                    Ks[i][j] = K_norm
                    dlambdas[i][j] = dlambda_norm
                    dsigmas[i][j] = dsigma_norm
                else:
                    Ks[i][j] = K_result
                    dlambdas[i][j] = dlambda
                    dsigmas[i][j] = dsigma

        return (Ks, dlambdas, dsigmas)

    cdef tuple _normalize(self, double K_result, double dlambda, double dsigma, double diag_Ks_i, 
                   double diag_Ks_j, double diag_dlambdas_i, double diag_dlambdas_j, 
                   double diag_dsigmas_i, double diag_dsigmas_j):
        """
        Normalize the result from SSTK, including derivatives.
        """
        cdef double norm, sqrt_nrorm, K_norm, diff_lambda, dlambda_norm, diff_sigma, dsigma_norm
        
        norm = diag_Ks_i * diag_Ks_j
        sqrt_norm = sqrt(norm)
        K_norm = K_result / sqrt_norm
                
        diff_lambda = ((diag_dlambdas_i * diag_Ks_j) +
                       (diag_Ks_i * diag_dlambdas_j))
        diff_lambda /= 2 * norm
        dlambda_norm = ((dlambda / sqrt_norm) -
                        (K_norm * diff_lambda))
        
        diff_sigma = ((diag_dsigmas_i * diag_Ks_j) +
                      (diag_Ks_i * diag_dsigmas_j))
        diff_sigma /= 2 * norm
        dsigma_norm = ((dsigma / sqrt_norm) -
                       (K_norm * diff_sigma))
        return K_norm, dlambda_norm, dsigma_norm
        
    cdef tuple _diag_calculations(self, np.ndarray X):
        """
        Calculate the K(x,x) values (required for normalization)
        """
        cdef np.ndarray[np.double_t, ndim=1] K_vec,dlambda_vec, dsimga_vec
        cdef list nodes
        cdef dict dict
        cdef double K_result, dlambda, dsigma
          
        K_vec = np.zeros(shape=(len(X),))
        dlambda_vec = np.zeros(shape=(len(X),))
        dsigma_vec = np.zeros(shape=(len(X),))
        for i, x in enumerate(X):
            nodes, dicts = self._tree_cache[x[0]]
            K_result, dlambda, dsigma = self._calc_K(nodes, nodes, dicts, dicts)
            K_vec[i] = K_result
            dlambda_vec[i] = dlambda
            dsigma_vec[i] = dsigma
        return (K_vec, dlambda_vec, dsigma_vec)



    cdef list _get_node_pairs(self,list nodes1, list nodes2):
        """
        The node pair detection method devised by Moschitti (2006).
        Fast way to determine which nodes should be compared
        """
        cdef list node_pairs = []
        cdef int i1 = 0
        cdef int i2 = 0
        cdef int reset2
        cdef Node n1, n2
        while i1 < len(nodes1) and i2 < len(nodes2):
            n1 = nodes1[i1]
            n2 = nodes2[i2]
            if n1.production > n2.production:
                i2 += 1
            elif n1.production < n2.production:
                i1 += 1
            else:
                while n1.production == n2.production:
                    reset2 = i2
                    while n1.production == n2.production:
                        node_pairs.append((n1, n2))
                        i2 += 1
                        if i2 >= len(nodes2):
                            break
                        n2 = nodes2[i2]
                    i1 += 1
                    if i1 >= len(nodes1):
                        break
                    i2 = reset2
                    n1 = nodes1[i1]
                    n2 = nodes2[i2]
        return node_pairs


    
    cdef tuple _delta(self, Node node1, Node node2, dict dict1, dict dict2,
               delta_matrix, dlambda_matrix, dsigma_matrix,
               double _lambda,  double _sigma):
        """
        Recursive method used in kernel calculation.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef int id1, id2, ch1, ch2, i
        cdef double val, prod, K_result, dlambda, dsigma, sum_lambda, sum_sigma, denom
        cdef double delta_result, dlambda_result, dsigma_result

        cdef Node n1, n2



        



        id1 = node1.node_id
        id2 = node2.node_id
        tup = (id1, id2)
        # check if already made this comparrision
        # then just read from memory
        val = delta_matrix[tup]
        if val > 0:
            return val, dlambda_matrix[tup], dsigma_matrix[tup]

        #we follow iterative scheme laid out in https://www.aclweb.org/anthology/Q15-1033.pdf (Beck 2015)


        if node1.children_ids == None:
            delta_matrix[tup] = _lambda
            dlambda_matrix[tup] = 1
            return (_lambda, 1, 0)


        prod = 1
        sum_lambda = 0
        sum_sigma = 0
        children1 = node1.children_ids
        children2 = node2.children_ids
        for i in range(len(children1)):
            ch1 = children1[i]
            ch2 = children2[i]
            n1 = dict1[ch1]
            n2 = dict2[ch2]
            if n1.production == n2.production:
                K_result, dlambda, dsigma = self._delta(n1, n2, dict1, dict2, 
                                                       delta_matrix,
                                                       dlambda_matrix,
                                                       dsigma_matrix,
                                                       _lambda, _sigma)
                denom = _sigma + K_result
                prod *= denom
                sum_lambda += dlambda / denom
                sum_sigma += (1 + dsigma) / denom
            else:
                prod *= _sigma
                sum_sigma += 1 /_sigma

        delta_result = _lambda * prod
        dlambda_result = prod + (delta_result * sum_lambda)
        dsigma_result = delta_result * sum_sigma

        delta_matrix[tup] = delta_result
        dlambda_matrix[tup] = dlambda_result
        dsigma_matrix[tup] = dsigma_result
        return (delta_result, dlambda_result, dsigma_result)


    cdef tuple _calc_K(self, list nodes1,list nodes2,dict dict1,dict dict2):
        """
        The actual SSTK kernel, evaluated over two node lists.
        It also calculates the derivatives wrt lambda and sigma.
        """
        cdef double K_total = 0
        cdef double dlambda_total = 0
        cdef double dsigma_total = 0
        cdef double K_result, dlambda, dsigma
        
        
        # We store the hypers inside C doubles and pass them as 
        # parameters for "delta", for efficiency.
        cdef double _lambda = self._lambda
        cdef double _sigma = self._sigma


        # Initialize the DP structure.
        delta_matrix = defaultdict(float)
        dlambda_matrix = defaultdict(float)
        dsigma_matrix = defaultdict(float)

        # only calculate over a subset of node pairs for efficiency
        node_pairs = self._get_node_pairs(nodes1, nodes2)
        for node_pair in node_pairs:
            K_result, dlambda, dsigma = self._delta(node_pair[0], node_pair[1], dict1, dict2,
                                                   delta_matrix, dlambda_matrix, dsigma_matrix,
                                                   _lambda, _sigma)
            K_total += K_result
            dlambda_total += dlambda
            dsigma_total += dsigma
        return (K_total, dlambda_total, dsigma_total)

    