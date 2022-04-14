import itertools
from typing import Iterable, Union, Tuple, List
import numpy as np
from emukit.core.parameter import Parameter


class CFGParameter(Parameter):
    """
    A class for inputs consisting of a string that can be generated from a context free grammar (CFG)
    """
    def __init__(self, name: str, grammar: object, max_length: int=15, min_length: int=1, cfactor: float=0.7):
        """
        :param name: Name of parameter
        :param grammar: cfg inducing the set of possible strings

        These other parameters control how we sample from the grammar (see self.sample_uniform)
        :max_length: maximum length of sampled tree from grammar. length means number of terminals
        :min_length: minimum length of sampled tree from grammar
        :cfactor: smaller cfactor provides smaller sequences (on average)
        """
        self.name = name
        self.grammar = grammar
        self.max_length = max_length
        self.min_length = min_length
        self.cfactor = cfactor
        self.alphabet = self.grammar.terminals

    def sample_uniform(self, point_count: int=1) -> np.ndarray:
        """
        Generates multiple (unqiue) random strings from the grammar
        :param point_count: number of data points to generate.
        :max_depth: maximum depth of sampled tree from grammar
        :max_num_prod: maximum number of productions in sampled trees
         
         
        :min_depth: minimum depth of sampled tree from grammar
        :min_num_prod: minimum number of productions in sampled trees
        :cfactor: smaller cfactor provides smaller sequences (on average)
        
        Sampling is following the algorithm described at 
        https://eli.thegreenplace.net/2010/01/28/generating-random-sentences-from-a-context-free-236grammar
        cfactor downweights frequent productions when traversing the grammar,
        ensuring that we do not hit python's max recursion depth
        with cfactor, this will happen as the longer the sequence, the less likely it is to end

        :returns: Generated points with shape (point_count, 1)
        """

        
        samples=self.grammar.sampler_restricted(point_count,self.max_length,self.cfactor,self.min_length)
        return np.array(samples).reshape(-1,1)

# helper function to go from parse trees to string
def unparse(tree):
    # turn tree into raw string form (as used for string kernel)
    string=[]
    temp=""
    # perform single pass of tree
    for char in tree:
        if char==" ":
            temp=""
        elif char==")":
            if temp[-1]!= ")":
                string.append(temp)
            temp+=char
        else:
            temp+=char
    return " ".join(string)
unparse = np.vectorize(unparse) 
