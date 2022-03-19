"""
python train.py selfies_vae 
"""

import argparse
import os
import sys
import torch
import rdkit
import sys 
sys.path.append('.')


from utils.script_utils import add_train_args, set_seed
from models.models_storage import ModelsStorage
from tdc.generation import MolGen
from models.selfies_vae import VAE, VAETrainer, vae_parser
from tdc.chem_utils import MolConvert
converter = MolConvert(src = 'SMILES', dst = 'SELFIES')

lg = rdkit.RDLogger.logger()
lg.setLevel(rdkit.RDLogger.CRITICAL)

MODELS = ModelsStorage()


def get_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title='Models trainer script', description='available models'
    )
    for model in MODELS.get_model_names():
        add_train_args(
            MODELS.get_model_train_parser(model)(
                subparsers.add_parser(model)
            )
        )
    return parser


def main(model, config):
    # model = 'selfies_vae'
    set_seed(config.seed)
    device = torch.device(config.device)

    if not os.path.exists(config.checkpoint_dir):
        os.mkdir(config.checkpoint_dir)

    config.config_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_config.pt')
    config.model_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_model.pt')
    config.vocab_save = os.path.join(config.checkpoint_dir, model + config.experimental_stuff + '_vocab.txt')

    if config.config_save is not None:
        torch.save(config, config.config_save)

    # For CUDNN to work properly
    if device.type.startswith('cuda'):
        torch.cuda.set_device(device.index or 0)
    
    trainer = MODELS.get_model_trainer(model)(config)
    # trainer = VAETrainer 

    if config.processed_data:
        train_data = trainer.load_train_data()
        val_data = trainer.load_val_data()
    else:
        data = MolGen(name = 'zinc', path = config.data_path)
        split = data.get_split(method = 'random', seed = config.data_seed, frac = [0.8, 0.0, 0.2])

        train_smiles = split['train']['smiles'].tolist()[:]
        val_smiles = split['test']['smiles'].tolist()[:]

        train_data = converter(train_smiles)
        val_data = converter(val_smiles)


    vocab = trainer.get_vocabulary(train_data)
    # print(vocab.c2i)
    # exit() 

    if config.vocab_save is not None:
        trainer.save_vocabulary(vocab)

    model = VAE(vocab, config).to(device)
    trainer.fit(model, train_data, val_data)


    model = model.to('cpu')
    torch.save(model, config.model_save)


if __name__ == '__main__':
    parser = get_parser()
    config = parser.parse_args()
    model = sys.argv[1]
    # model = 'smiles_vae' ## smiles_vae
    main(model, config)



"""

[C][C][C][=C][C][=C][Branch2][Ring1][Ring2][C][=N][C][C][=Branch1][C][=O][NH1][C][Branch1][C][N][=N][C][=Ring1][Branch2][NH1][Ring1][O][O][Ring1][S]
[O][=C][Branch1][S][C][C][=C][C][=C][C][=C][C][=C][C][Ring1][=Branch1][=C][Ring1][#Branch2][N][N][C][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][Cl]
[C][C][C][=C][C][=C][C][=C][Ring1][=Branch1][N][Branch1][=Branch1][C][Branch1][C][C][=O][C][=N][C][Branch1][=N][C][S][C][=N][N][=C][Branch1][C][N][S][Ring1][=Branch1][=C][S][Ring1][=N]
[NH3+1][C][C][C][O][C][C][C][N][Branch1][O][C][C][NH1+1][C][C][C][C][C][Ring1][=Branch1][C][C][Ring1][=C]
[C][C][=C][C][Branch2][Ring2][Branch2][C][=Branch1][C][=O][N][C][C][C][Branch2][Ring1][Branch2][C][N][C][=Branch1][C][=O][C][=C][C][=Branch1][C][=O][C][=C][C][=C][C][=C][Ring1][=Branch1][O][Ring1][O][C][C][Ring2][Ring1][Branch1][=N][O][Ring2][Ring1][N]
[C][C@@H1][C][C@@H1][Branch1][C][C][C][N][Branch2][Ring1][Branch1][C][=Branch1][C][=O][C][S][C][=C][S][C][Branch1][=Branch1][C][=Branch1][C][=O][O-1][=C][Ring1][Branch2][C][Ring2][Ring1][Ring1]
[C][C][=C][C][=C][C][=C][Branch2][Ring1][O][C][=C][Ring1][=Branch1][N][=C][Ring1][#Branch2][S][C][C][=Branch1][C][=O][N][C@@H1][Branch1][C][C][C][C][C][C@@H1][Ring1][#Branch1][C][O][C][C][O][Ring2][Ring1][=Branch1]
[C][O][C][=Branch1][C][=O][C][=C][Branch1][=C][N][C][=Branch1][C][=O][C][S][C][=N][C][=Ring1][Branch1][C][S][C][Branch1][C][C][=C][Ring1][#C][C]
[C][N][C][=N][C][=N][C][Branch1][=C][N][C][=C][C][=C][C][Branch1][C][Cl][=C][Ring1][#Branch1][C][=C][Ring1][#C][N]
[C][O][C][=C][C][=C][Branch2][Ring1][#Branch2][N][C][=C][Branch1][C][Cl][C][=Branch1][C][=O][N][Branch1][=Branch2][C][=C][C][=C][C][=C][Ring1][=Branch1][C][Ring1][=N][=O][C][Branch1][Ring1][O][C][=C][Ring2][Ring1][#Branch1]
[C][O][C][=Branch1][C][=O][N][C][C][C][Branch2][Ring1][Ring2][C][N][C][=Branch1][C][=O][C][=N][C][=C][C][=C][C][=C][Ring1][=Branch1][S][Ring1][=Branch2][C][C][Ring2][Ring1][Ring1]
[C][NH2+1][C@@H1][Branch2][Ring1][Branch2][C][C][=C][N][Branch1][#C][C][C][=Branch1][C][=O][N][C][C][C][O][C][C][Ring1][=Branch1][C][=Ring1][#C][C][C][C][Ring1][Ring1]
[C][C@@H1][Branch1][=Branch2][C][=C][C][=C][C][=N][Ring1][=Branch1][NH1+1][Branch1][C][C][C][C][=Branch1][C][=O][N][C][=N][N][=C][Branch1][=Branch2][C][Branch1][C][C][Branch1][C][C][C][S][Ring1][=Branch2]





"""


