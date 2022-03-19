import argparse


def get_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    # Model
    model_arg = parser.add_argument_group('Model')
    model_arg.add_argument('--q_bidir',
                           default=False, action='store_true',
                           help='If to add second direction to encoder')
    model_arg.add_argument('--hidden_size',
                           type=int, default=450,
                           help='Hidden dimensionality')
    model_arg.add_argument('--batch_size',
                           type=int, default=32,
                           help='Number of samples in a batch')
    model_arg.add_argument('--latent_size',
                           type=int, default=56,
                           help='Dimension of latent space')
    model_arg.add_argument('--depthT',
                           type=int, default=20,
                           help='Depth of T')
    model_arg.add_argument('--depthG',
                           type=int, default=3,
                           help='Depth of G')

    # Train
    train_arg = parser.add_argument_group('Train')
    model_arg.add_argument('--lr',
                           type=float, default=1e-3,
                           help='Initial learning rate')
    model_arg.add_argument('--clip_norm',
                           type=float, default=50,
                           help='Maximum norm of derivative')
    model_arg.add_argument('--beta',
                           type=float, default=0.0,
                           help='Initial Beta')
    model_arg.add_argument('--steo_beta',
                           type=float, default=0.002,
                           help='Step Beta')
    model_arg.add_argument('--max_beta',
                           type=float, default=1.0,
                           help='Maximum Beta')
    model_arg.add_argument('--warmup',
                           type=int, default=40000,
                           help='Warmup')
    model_arg.add_argument('--epoch',
                           type=int, default=20,
                           help='Initial learning rate')
    model_arg.add_argument('--load_epoch',
                           type=int, default=0,
                           help='Load previous training epoches')
    model_arg.add_argument('--anneal_rate',
                           type=float, default=0.9,
                           help='Initial learning rate')
    model_arg.add_argument('--anneal_iter',
                           type=int, default=40000,
                           help='Initial learning rate')
    model_arg.add_argument('--kl_anneal_iter',
                           type=int, default=2000,
                           help='Initial learning rate')
    model_arg.add_argument('--print_iter',
                           type=int, default=50,
                           help='Initial learning rate')
    model_arg.add_argument('--save_iter',
                           type=int, default=5000,
                           help='Initial learning rate')

    return parser

