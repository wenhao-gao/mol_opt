"""
Filters the synthetic trees by the QEDs of the root molecules.
"""
from tdc import Oracle
qed = Oracle(name='qed')
import numpy as np
import pandas as pd
from syn_net.utils.data_utils import *

def is_valid(smi):
    """
    Checks if a SMILES string is valid.

    Args:
        smi (str): Molecular SMILES string.

    Returns:
        False or str: False if the SMILES is not valid, or the reconverted
            SMILES string.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return False
    else:
        return Chem.MolToSmiles(mol, isomericSmiles=False)

if __name__ == '__main__':

    data_path = '/pool001/whgao/data/synth_net/st_pis/st_data.json.gz'
    st_set = SyntheticTreeSet()
    st_set.load(data_path)
    data = st_set.sts
    print(f'Finish reading, in total {len(data)} synthetic trees.')

    filtered_data = []
    original_qed = []
    qeds = []
    generated_smiles = []

    threshold = 0.5

    for t in tqdm(data):
        try:
            valid_smiles = is_valid(t.root.smiles)
            if valid_smiles:
                if valid_smiles in generated_smiles:
                    pass
                else:
                    qed_value = qed(valid_smiles)
                    original_qed.append(qed_value)

                    # filter the trees based on their QEDs
                    if qed_value > threshold or np.random.random() < (qed_value/threshold):
                        generated_smiles.append(valid_smiles)
                        filtered_data.append(t)
                        qeds.append(qed_value)
                    else:
                        pass
            else:
                pass
        except:
            pass

    print(f'Finish sampling, remaining {len(filtered_data)} synthetic trees.')

    st_set = SyntheticTreeSet(filtered_data)
    st_set.save('/pool001/whgao/data/synth_net/st_pis/st_data_filtered.json.gz')

    df = pd.DataFrame({'SMILES': generated_smiles, 'qed': qeds})
    df.to_csv('/pool001/whgao/data/synth_net/st_pis/filtered_smiles.csv.gz', compression='gzip', index=False)

    print('Finish!')
