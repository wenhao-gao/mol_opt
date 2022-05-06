"""
Reaction network.
"""
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from syn_net.models.mlp import MLP, load_array
from scipy import sparse


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--featurize", type=str, default='fp',
                        help="Choose from ['fp', 'gin']")
    parser.add_argument("-r", "--rxn_template", type=str, default='hb',
                        help="Choose from ['hb', 'pis']")
    parser.add_argument("--radius", type=int, default=2,
                        help="Radius for Morgan fingerprint.")
    parser.add_argument("--nbits", type=int, default=4096,
                        help="Number of Bits for Morgan fingerprint.")
    parser.add_argument("--out_dim", type=int, default=300,
                        help="Output dimension.")
    parser.add_argument("--ncpu", type=int, default=8,
                        help="Number of cpus")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--epoch", type=int, default=2000,
                        help="Maximum number of epochs.")
    parser.add_argument("--restart", type=bool, default=False,
                        help="Indicates whether to restart training.")
    parser.add_argument("-v", "--version", type=int, default=1,
                        help="Version")
    args = parser.parse_args()

    if args.out_dim == 300:
        validation_option = 'nn_accuracy_gin'
    elif args.out_dim == 4096:
        validation_option = 'nn_accuracy_fp_4096'
    elif args.out_dim == 256:
        validation_option = 'nn_accuracy_fp_256'
    elif args.out_dim == 200:
        validation_option = 'nn_accuracy_rdkit2d'
    else:
        raise ValueError

    main_dir   = f'/pool001/whgao/data/synth_net/{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_{validation_option[12:]}/'
    batch_size = args.batch_size
    ncpu       = args.ncpu

    X = sparse.load_npz(main_dir + 'X_rxn_train.npz')
    y = sparse.load_npz(main_dir + 'y_rxn_train.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    train_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=True)

    X = sparse.load_npz(main_dir + 'X_rxn_valid.npz')
    y = sparse.load_npz(main_dir + 'y_rxn_valid.npz')
    X = torch.Tensor(X.A)
    y = torch.LongTensor(y.A.reshape(-1, ))
    valid_data_iter = load_array((X, y), batch_size, ncpu=ncpu, is_train=False)

    pl.seed_everything(0)
    param_path  = f'/pool001/rociomer/data/pre-trained-models/{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_v{args.version}/'
    path_to_rxn = f'{param_path}rxn.ckpt'
    if not args.restart:

        if args.featurize == 'fp':
            if args.rxn_template == 'hb':
                mlp = MLP(input_dim=int(4 * args.nbits),
                          output_dim=91,
                          hidden_dim=3000,
                          num_layers=5,
                          dropout=0.5,
                          num_dropout_layers=1,
                          task='classification',
                          loss='cross_entropy',
                          valid_loss='accuracy',
                          optimizer='adam',
                          learning_rate=1e-4,
                          val_freq=10,
                          ncpu=ncpu)
            elif args.rxn_template == 'pis':
                mlp = MLP(input_dim=int(4 * args.nbits),
                          output_dim=4700,
                          hidden_dim=4500,
                          num_layers=5,
                          dropout=0.5,
                          num_dropout_layers=1,
                          task='classification',
                          loss='cross_entropy',
                          valid_loss='accuracy',
                          optimizer='adam',
                          learning_rate=1e-4,
                          val_freq=10,
                          ncpu=ncpu)
        elif args.featurize == 'gin':
            if args.rxn_template == 'hb':
                mlp = MLP(input_dim=int(3 * args.nbits + args.out_dim),
                          output_dim=91,
                          hidden_dim=3000,
                          num_layers=5,
                          dropout=0.5,
                          num_dropout_layers=1,
                          task='classification',
                          loss='cross_entropy',
                          valid_loss='accuracy',
                          optimizer='adam',
                          learning_rate=1e-4,
                          val_freq=10,
                          ncpu=ncpu)
            elif args.rxn_template == 'pis':
                mlp = MLP(input_dim=int(3 * args.nbits + args.out_dim),
                          output_dim=4700,
                          hidden_dim=3000,
                          num_layers=5,
                          dropout=0.5,
                          num_dropout_layers=1,
                          task='classification',
                          loss='cross_entropy',
                          valid_loss='accuracy',
                          optimizer='adam',
                          learning_rate=1e-4,
                          val_freq=10,
                          ncpu=ncpu)
    else:
        if args.rxn_template == 'hb':
            mlp = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * args.nbits),
                output_dim=91,
                hidden_dim=3000,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )
        elif args.rxn_template == 'pis':
            mlp = MLP.load_from_checkpoint(
                path_to_rxn,
                input_dim=int(4 * args.nbits),
                output_dim=4700,
                hidden_dim=4500,
                num_layers=5,
                dropout=0.5,
                num_dropout_layers=1,
                task='classification',
                loss='cross_entropy',
                valid_loss='accuracy',
                optimizer='adam',
                learning_rate=1e-4,
                ncpu=ncpu
            )

    tb_logger = pl_loggers.TensorBoardLogger(f'rxn_{args.rxn_template}_{args.featurize}_{args.radius}_{args.nbits}_logs/')
    trainer   = pl.Trainer(gpus=[0], max_epochs=args.epoch, progress_bar_refresh_rate=20, logger=tb_logger)
    t         = time.time()

    trainer.fit(mlp, train_data_iter, valid_data_iter)

    print(time.time() - t, 's')

    print('Finish!')
