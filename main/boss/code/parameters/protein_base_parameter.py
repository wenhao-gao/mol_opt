import itertools
from typing import Iterable, Union, Tuple, List
import numpy as np
from emukit.core.parameter import Parameter


class ProteinBaseParameter(Parameter):
    """
    A class for a particular protein sequence (made from amino acids)
    The space is all synonymous sequences of genes representing this amino acid sequence (represented in terms of bases)
    """
    def __init__(self, name: str, sequence: str):
        """
        :param name: Name of parameter
        :Sequence: input gene
        """
        self.name = name
        self.sequence = sequence
        self.codon_table = create_codon_table()
        self.codon_to_bases = {str(value):key for key, value in create_codon_index().items()}
        self.acid_table = create_aminoacids_table()
        self.acid_to_codon_table = create_aminoacids_to_codon_index_table()
        # store a single gene representation of the amino acid (one of the many possible)
        self.example_gene_representation = convert(sequence, self.acid_table)
        self.length = len(self.sequence)
        # store possible swaps for each poistion (in terms of bases)
        self.possible_swaps = [self.acid_table[x] for x in self.sequence]
        # number of possible representations of the protein
        self.number_of_canidates = np.prod([len(x) for x in self.possible_swaps])
        self.alphabet=["a","c","t","g"]

    def sample_uniform(self, point_count: int=1) -> np.ndarray:
        """
        Generates multiple random gene representations of the amino acid
        :param point_count: number of data points to generate.
        :returns: Generated points with shape (point_count, 1)
        """
        samples = np.zeros((point_count,1),dtype=object)
        for i in range(0,point_count):
            sample = generate_random_sequence(self.example_gene_representation)
            samples[i][0]=" ".join(list(sample))
        return samples
        

# helper functions to deal with gene sequences

## define the table of codons/aminoacids
def create_codon_table():
    bases = ['t', 'c', 'a', 'g']
    codons = [a+b+c for a in bases for b in bases for c in bases]
    acids = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
    codon_table = dict(zip(codons, acids))
    return codon_table

## define the table of aminoacids
def create_aminoacids_table():
    codon_dict = create_codon_table()
    amin_dict = {}
    for k, v in codon_dict.items():
        amin_dict.setdefault(v, []).append(k)
    return(amin_dict)

# define table of codons (by index rather than bases)
def create_codon_index():
    bases = ['t', 'c', 'a', 'g']
    codons = [a+b+c for a in bases for b in bases for c in bases]
    indicies = range(len(codons))
    return dict(zip(codons,indicies))
    
## define the table of aminoacids (by index rather than bases)
def create_aminoacids_to_codon_index_table():
    codon_dict = create_codon_table()
    codon_index = create_codon_index()
    amin_dict = {}
    for k, v in codon_dict.items():
        amin_dict.setdefault(v, []).append(str(codon_index[k]))
    return(amin_dict)


## generates a random coherent sequence of bases from a sample sequence
def generate_random_sequence(seq):
    amin_dict = create_aminoacids_table()
    codon_dict = create_codon_table()
    trans_seq = translate(seq, codon_dict)
    Namin = len(trans_seq)
    new_seq = ''
    for k in range(Namin):
        redundant_codons = amin_dict[trans_seq[k]]
        Range = np.array(range(len(redundant_codons)))
        np.random.shuffle(Range)
        new_seq=new_seq +redundant_codons[Range[0]]
    return(new_seq)


# given the initial amino acid sequnce (e.g. "MGVHECPAWL")
# generate random representation in terms of codon index (i.e '35 60 49 25 58 13 20 55 15 18')
def generate_random_codon_sequence(seq):
    new_seq=[]
    amin_dict = create_aminoacids_to_codon_index_table()
    codon_dict = create_codon_index()
    for k in range(len(seq)):
        redundant_codons = amin_dict[seq[k]]
        Range = np.array(range(len(redundant_codons)))
        np.random.shuffle(Range)
        new_seq.append(str(redundant_codons[Range[0]]))
    return " ".join(new_seq)



## translate gene seqence (bases) to aminoacids sequence
def translate(seq, code):
    return "".join((code[seq[i:i+3]] for i in range(0, len(seq)-len(seq)%3, 3)))

## convert aminoacids sequnce to a gene (just one of many possible choices)
def convert(seq,code):
    return "".join((code[seq[i]][0] for i in range(0, len(seq))))
