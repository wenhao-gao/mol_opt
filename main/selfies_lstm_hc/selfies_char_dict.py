from tdc.chem_utils import MolConvert
smiles2selfies = MolConvert(src = 'SMILES', dst = 'SELFIES')
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')
import os 
path_here = os.path.dirname(os.path.realpath(__file__))


class SelfiesCharDictionary(object):
    """
    A fixed dictionary for druglike SMILES.
    Enables smile<->token conversion.

    With a space:0 for padding, Q:1 as the start token and end_of_line \n:2 as the stop token.
    """

    PAD = ' '
    BEGIN = 'Q'
    END = '\n'

    def __init__(self) -> None:

        self.forbidden_symbols = {}
        with open(os.path.join(path_here,'Voc'), 'r') as fin:
            word_list = fin.readlines() 
        word_list = [word.strip() for word in word_list]
        word_list = [self.PAD, self.BEGIN, self.END,] + word_list 
        self.char_idx = {word:idx for idx,word in enumerate(word_list)}

        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = dict() 
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def allowed(self, smiles) -> bool:
        """
        Determine if smiles string has illegal symbols

        Args:
            smiles: SMILES string

        Returns:
            True if all legal
        """
        return True

    def encode(self, smiles: str) -> str:
        """
        Replace multi-char tokens with single tokens in SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            sanitized SMILE string with only single-char tokens
        """
        words = smiles.strip().strip('[]').split('][')
        temp_smiles = ['['+word+']' for word in words]
        return temp_smiles

    def decode(self, smiles):
        """
        Replace special tokens with their multi-character equivalents.

        Args:
            smiles: SMILES string

        Returns:
            SMILES string with possibly multi-char tokens
        """
        temp_smiles = smiles
        return temp_smiles

    def get_char_num(self) -> int:
        """
        Returns:
            number of characters in the alphabet
        """
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[self.BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[self.END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[self.PAD]

    def matrix_to_smiles(self, array):
        """
        Converts an matrix of indices into their SMILES representations

        Args:
            array: torch tensor of indices, one molecule per row

        Returns: a list of SMILES, without the termination symbol
        """
        smiles_strings = []

        for row in array:
            predicted_chars = []

            for j in row:
                next_char = self.idx_char[j.item()]
                if next_char == self.END:
                    break
                predicted_chars.append(next_char)

            smi = ''.join(predicted_chars)
            smi = self.decode(smi)
            smiles_strings.append(smi)

        return smiles_strings





