"""
Reads synthetic tree data and splits it into training, validation and testing sets.
"""
from syn_net.utils.data_utils import *


if __name__ == "__main__":

    st_set = SyntheticTreeSet()
    path_to_data = '/pool001/whgao/data/synth_net/st_pis/st_data_filtered.json.gz'
    print('Reading data from ', path_to_data)
    st_set.load(path_to_data)
    data = st_set.sts
    del st_set
    num_total = len(data)
    print("In total we have: ", num_total, "paths.")

    split_ratio = [0.6, 0.2, 0.2]

    num_train = int(split_ratio[0] * num_total)
    num_valid = int(split_ratio[1] * num_total)
    num_test = num_total - num_train - num_valid

    data_train = data[:num_train]
    data_valid = data[num_train: num_train + num_valid]
    data_test = data[num_train + num_valid: ]

    print("Saving training dataset: ", len(data_train))
    tree_set = SyntheticTreeSet(data_train)
    tree_set.save('/pool001/whgao/data/synth_net/st_pis/st_train.json.gz')

    print("Saving validation dataset: ", len(data_valid))
    tree_set = SyntheticTreeSet(data_valid)
    tree_set.save('/pool001/whgao/data/synth_net/st_pis/st_valid.json.gz')

    print("Saving testing dataset: ", len(data_test))
    tree_set = SyntheticTreeSet(data_test)
    tree_set.save('/pool001/whgao/data/synth_net/st_pis/st_test.json.gz')

    print("Finish!")
