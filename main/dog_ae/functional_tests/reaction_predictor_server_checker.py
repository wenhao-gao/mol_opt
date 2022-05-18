"""
Check that the OpenNMT reaction predictor interaction works okay by eyeballing the output.

CUDA_VISIBLE_DEVICES="0,1" python server.py --config available_models/mtransformer_example_server.conf.json
CUDA_VISIBLE_DEVICES=0 python server.py --config available_models/mtransformer_example_server.conf.json

with the json config looking like:
{
  "models_root": "./saved_models/uspto_no_reagents/",
  "models": [
    {
      "id": 0,
      "model": "uspto_no_reagents_model_step_250000.pt",
      "timeout": 600,
      "on_timeout": "to_cpu",
      "load": true,
      "opt": {
        "gpu": 1,
        "replace_unk": true,
        "max_length": 500,
        "fast": true,
        "n_best": 1
      }
    }
  ]
}

out [Multiset({'CSc1ccc(Cl)cc1': 1}), Multiset({'CC(C)(C)OC(=O)N1CCC(C(=O)N2CCN(C(=O)OC(C)(C)C)CC2)CC1': 1})]

"""

import multiset

from syn_dags.model import reaction_predictors

def main():
    rp = reaction_predictors.OpenNMTServerPredictor()
    reactants = [
        multiset.FrozenMultiset(['C[S-]', '[Mg+]c1ccc(Cl)cc1']),
        multiset.FrozenMultiset(['CCOC(=O)C1CCN(C(=O)OC(C)(C)C)CC1', 'CC(C)(C)OC(=O)N1CCNCC1'])
    ]
    print(rp(reactants))
    print(rp(reactants))
    print(rp(reactants))


if __name__ == '__main__':
    main()
