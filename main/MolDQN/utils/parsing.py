"""
Code handle the arguments
"""

from argparse import ArgumentParser, Namespace
import json


def add_args(parser: ArgumentParser):
    """
    Adds training arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    # General arguments
    parser.add_argument('-p', '--parameters', default=None,
                        help='The network parameters to begin with.')
    parser.add_argument('-t', '--task', default='test',
                        help='The task name.')
    parser.add_argument('-c', '--path_to_config', default=None,
                        help='The JSON file define the hyper parameters.')
    parser.add_argument('-i', '--init_mol', default='C',
                        help='The initial molecule to start with.')
    parser.add_argument('-m', '--model_path', default='./checkpoints/',
                        help='path to put output files.')
    parser.add_argument('-o', '--objective', default='mpo_zaleplon', nargs='+',
                        help='The objective to optimize')
    parser.add_argument('-n', '--number', default=100,
                        help='Number of molecules to keep')
    parser.add_argument('-v', '--verbose', action='store_true', default=True,
                        help='Whether to see the intermediate information.')
    parser.add_argument('--save_frequency', default=1000,
                        help='The frequency to save a checkpoint.')

    # Molecular environment arguments
    parser.add_argument('--atom_types', default=['C', 'O', 'N'],
                        help='The atom type allowed to use.')
    parser.add_argument('--max_steps_per_episode', default=2,
                        help='The steps in one episode.')
    parser.add_argument('--allow_removal', action='store_true', default=False,
                        help='Allow removal steps or not.')
    parser.add_argument('--allow_no_modification', action='store_true', default=True,
                        help='Allow no modification steps or not.')
    parser.add_argument('--allow_bonds_between_rings', action='store_true', default=False,
                        help='Allow forming bonds between rings or not.')
    parser.add_argument('--allowed_ring_sizes', default=[3, 4, 5, 6, 7],
                        help='Allowed ring sizes.')
    parser.add_argument('--discount_factor', default=1,
                        help='The discount factor of reward.')
    parser.add_argument('--synthesizability', default=None,
                        choices=[None, 'sa', 'sc', 'smi'],
                        help='Use what kind of synthesizability to bias the training.')

    # Reinforcement Learning arguments
    parser.add_argument('--num_episodes', default=10,
                        help='Number of episodes to run.')
    parser.add_argument('--replay_buffer_size', default=5000,
                        help='The action reward sets to store in the replay buffer.')
    parser.add_argument('--batch_size', default=128,
                        help='The training batch size.')
    parser.add_argument('--double_q', action='store_true',
                        help='Use double dqn setting or not.')
    parser.add_argument('--gamma', default=1,
                        help='The normally defined discount factor.')
    parser.add_argument('--update_frequency', default=20,
                        help='The target network update frequency.')
    parser.add_argument('--num_bootstrap_heads', default=36,
                        help='The number of bootstrap heads.')
    parser.add_argument('--prioritized', action='store_true', default=True,
                        help='Use the prioritized replay or not.')
    parser.add_argument('--prioritized_alpha', default=0.6,
                        help='The alpha parameter in prioritized replay.')
    parser.add_argument('--prioritized_beta', default=0.4,
                        help='The beta parameter in prioritized replay.')
    parser.add_argument('--prioritized_epsilon', default=0.000001,
                        help='A small number to avoid zero denominator.')
    parser.add_argument('--exploration', default='bootstrap',
                        choices=['bootstrap', 'disagreement'],
                        help='Choose the exploration method.')

    # Q function arguments
    parser.add_argument('--q_function', default='mlp',
                        choices=['mpnn', 'mlp'],
                        help='The q funciton.')
    parser.add_argument('--noisy', action='store_true', default=False,
                        help='Whether use noisy layer.')
    parser.add_argument('--distribution', action='store_true', default=False,
                        help='Whether to use distributional DQN.')
    parser.add_argument('--activation', type=str, default='ReLU',
                        choices=['ReLU', 'LeakyReLU', 'PReLU', 'tanh', 'SELU', 'ELU'],
                        help='Activation function')
    parser.add_argument('--optimizer', default='Adam',
                        help='The opitmizer to use.')
    parser.add_argument('--batch_norm', action='store_true', default=False,
                        help='Use the batch normalization or not.')
    parser.add_argument('--dropout', default=0,
                        help='The dropout probability.')
    parser.add_argument('--adam_beta_1', default=0.9,
                        help='The beta_1 in adam optimizer.')
    parser.add_argument('--adam_beta_2', default=0.999,
                        help='The beta_2 in adam optimizer.')
    parser.add_argument('--grad_clipping', default=10,
                        help='The gradient clipping.')
    parser.add_argument('--learning_frequency', default=4,
                        help='The frequency of learning, steps')
    parser.add_argument('--learning_rate', default=1e-4,
                        help='The learning rate to begin with.')
    parser.add_argument('--learning_rate_decay_steps', default=1000,
                        help='Learning rate decay steps.')
    parser.add_argument('--learning_rate_decay_rate', default=0.8,
                        help='Learning rate decay rate.')

    # Morgan FFN part
    parser.add_argument('--fingerprint_radius', default=3,
                        help='The Morgan fingerprint radius.')
    parser.add_argument('--fingerprint_length', default=2048,
                        help='The Morgan fingerprint length.')
    parser.add_argument('--dense_layers', default=[1024, 512, 128, 32],
                        help='The dense layers of ffn.')

    # Graph network part
    parser.add_argument('--hidden_size', default=300,
                        help='The hidden vector size.')
    parser.add_argument('--bias', action='store_true', default=False,
                        help='To use bias in graph network or not.')
    parser.add_argument('--depth', default=3,
                        help='The message passing depth of graph network.')
    parser.add_argument('--ffn_hidden_size', default=300,
                        help='The hidden size of following ffn.')
    parser.add_argument('--ffn_num_layers', default=2,
                        help='The number of layers of following ffn.')
    parser.add_argument('--atom_messages', action='store_true', default=False,
                        help='To use atom features instead of concatenation of atom and bond')
    parser.add_argument('--feature_only', action='store_true', default=True,
                        help='Only use the artificial features.')
    parser.add_argument('--use_input_features', action='store_true', default=True,
                        help='Concatenate input features.')
    parser.add_argument('--features_dim', default=1,
                        help='The feature dimension.')
    parser.add_argument('--no_cache', action='store_true', default=False,
                        help='Turn off caching mol2graph computation.')


def modify_args(args: Namespace):
    """Modify the arguments and read json configuration file to overwrite."""
    hparams = {}
    if args.path_to_config is not None:
        with open(args.path_to_config, 'r') as f:
            hparams.update(json.load(f))

        for key, value in hparams.items():
            setattr(args, key, value)
    return args


def parse_args() -> Namespace:
    """
    Parses arguments for training (includes modifying/validating arguments).

    :return: A Namespace containing the parsed, modified, and validated args.
    """
    parser = ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    modify_args(args)

    return args


if __name__ == "__main__":
    args = parse_args()
    print(type(args.atom_types))
