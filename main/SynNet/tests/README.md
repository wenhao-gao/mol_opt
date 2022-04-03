# Unit tests

## Instructions
To run the unit tests, start from the main SynNet directory and run:

```
export PYTHONPATH=`pwd`:$PYTHONPATH
```

Then, activate the SynNet conda environment, and from the current unit tests directory, run:

```
python -m unittest
```

## Dataset
The data used for unit testing consists of:
* 3 randomly sampled reaction templates from the Hartenfeller-Button dataset (*rxn_set_hb_test.txt*)
* 100 randomly sampled matching building blocks from Enamine (*building_blocks_matched.csv.gz*)
