"""
Reads synthetic tree data and prints the first five trees.
"""
from syn_net.utils.data_utils import *


if __name__ == "__main__":

    st_set = SyntheticTreeSet()
    path_to_data = '/pool001/whgao/data/synth_net/st_pis/st_data.json.gz'

    print('Reading data from ', path_to_data)
    st_set.load(path_to_data)
    data = st_set.sts

    for t in data[:5]:
        t._print()

    print(len(data))
    print("Finish!")
