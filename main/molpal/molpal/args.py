from configargparse import ArgumentTypeError, ArgumentParser, Namespace
from pathlib import Path
from typing import Optional, Union


def gen_args(args: Optional[str] = None) -> Namespace:
    parser = ArgumentParser()

    add_general_args(parser)
    add_encoder_args(parser)
    add_pool_args(parser)
    add_acquisition_args(parser)
    add_objective_args(parser)
    add_model_args(parser)
    add_stopping_args(parser)

    args = parser.parse_args(args)

    cleanup_args(args)

    return args

#################################
#       GENERAL ARGUMENTS       #
#################################
def add_general_args(parser: ArgumentParser) -> None:
    parser.add_argument('method', default='molpal')
    parser.add_argument('--config', is_config_file=True,
                        help='the filepath of the configuration file')
    parser.add_argument('--output-dir', default="molpal_out",
                        help='the name of the output directory')
    parser.add_argument('--wandb', type=str, default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument('--seed', type=int,
                        help='the random seed to use for initialization.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='the level of output this program should print')
    parser.add_argument('-nc', '--ncpu', '--cpus', default=1, type=int, 
                        help='the number of cores to available to each GPU for dataloading purposes. If performing docking, this is also the number of cores multithreaded docking programs will utilize.')
    parser.add_argument('--write-intermediate', 
                        action='store_true', default=False,
                        help='whether to write a summary file with all of the explored inputs and their associated scores after each round of exploration')
    parser.add_argument('--write-final', action='store_true', default=False,
                        help='whether to write a summary file with all of the explored inputs and their associated scores')

    parser.add_argument('--chkpt-freq', type=int,
                        nargs='?', default=99999999999, const=-1,
                        help='The number of iterations that should pass without writing a checkpoint. A value of 0 means writing a checkpoint file every iteration. A value of 1 corresponds to skipping one iteration between checkpoints and so on. A value of -1 or below will result in no checkpointing. By default, checkpointing occurs every iteration. For convenience, passing solely the flag with no argument will result in no checkpointing at all.')
    parser.add_argument('--checkpoint-file',
                        help='the checkpoint file containing the state of a previous molpal run.')
    parser.add_argument('--previous-scores',
                        help='the path to a file containing the scores from a previous run of molpal to load in as preliminary dataset.')
    parser.add_argument('--scores-csvs', nargs='+',
                        help='Either (1) A list of filepaths containing the outputs from a previous exploration or (2) a pickle file containing this list. Will load these files in the order in which they are passed to mimic the intermediate state of a previous exploration. Specifying a single will be interpreted as passing a pickle file. If seeking to mimic the state after only one round of exploration, use the --previous-scores argument instead and leave this option empty.')
    parser.add_argument('--log_results', action='store_true')
    parser.add_argument('--log_code', action='store_true')
    parser.add_argument('--smi_file', default=None)
    parser.add_argument('--config_default', default='hparams_default.yaml')
    parser.add_argument('--config_tune', default='hparams_tune.yaml')
    parser.add_argument('--pickle_directory', help='Directory containing pickle files with the distribution statistics', default=None)
    parser.add_argument('--n_jobs', type=int, default=-1)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_oracle_calls', type=int, default=10000)
    parser.add_argument('--freq_log', type=int, default=100)
    parser.add_argument('--n_runs', type=int, default=5)
    parser.add_argument('--task', type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument('--oracles', nargs="+", default=["QED"])

#####################################
#       ENCODER ARGUMENTS           #
#####################################
def add_encoder_args(parser: ArgumentParser) -> None:
    parser.add_argument('--fingerprint', default='pair',
                        choices={'morgan', 'rdkit', 'pair', 'maccs', 'map4'},
                        help='the type of encoder to use')
    parser.add_argument('--radius', type=int, default=2,
                        help='the radius or path length to use for fingerprints')
    parser.add_argument('--length', type=int, default=2048,
                        help='the length of the fingerprint')

##############################
#       POOL ARGUMENTS       #
##############################
def add_pool_args(parser: ArgumentParser) -> None:
    parser.add_argument('--pool', default='eager',
                        help='the type of MoleculePool to use')
    parser.add_argument('-l', '--libraries', '--library',
                        default=['data/zinc.csv.gz'], nargs='+',
                        help='the CSVs containing members of the MoleculePool')
    parser.add_argument('--no-title-line', action='store_true', default=False,
                        help='whether there is no title line in the library files')
    parser.add_argument('--delimiter', default=',',
                        help='the column separator in the library files')
    parser.add_argument('--smiles-col', default=0, type=int,
                        help='the column containing the SMILES string in the library files')
    parser.add_argument('--cxsmiles', default=False, action='store_true',
                        help='whether the files use CXSMILES strings')
    parser.add_argument('--fps', metavar='FPS_HDF5',
                        help='an HDF5 file containing the precalculated feature representation of each molecule in the pool')
    parser.add_argument('--cluster', action='store_true', default=False,
                        help='whether to cluster the MoleculePool')
    parser.add_argument('--cache', action='store_true', default=False,
                        help='whether to store the SMILES strings of the MoleculePool in memory')
    parser.add_argument('--invalid-idxs', '--invalid-lines',
                        type=int, nargs='*',
                        help='the indices in the overall library (potentially consisting of multiple library files) containing invalid SMILES strings')

#####################################
#       ACQUISITION ARGUMENTS       #
#####################################
def add_acquisition_args(parser: ArgumentParser) -> None:
    parser.add_argument('--metric', '--alpha', default='random',
                        choices={'random', 'greedy', 'threshold', 'ts',
                                 'ucb', 'ei', 'pi', 'thompson'},
                        help='the acquisition metric to use')

    parser.add_argument('--init-size',
                        type=restricted_float_or_int, default=0.01,
                        help='the number of ligands or fraction of total library to initially sample')
    parser.add_argument('--batch-sizes',
                        type=restricted_float_or_int, default=[0.01], nargs='+',
                        help='the number of ligands or fraction of total library to sample for each successive batch of exploration. Will proceed through the values provided and repeat the final value as neccessary. I.e., passing --batch-sizes 10 20 30 will acquire 10 inputs in the first exploration iteration, 20 in the second, and 30 for all remaining exploration batches')
    parser.add_argument('--epsilon', type=float, default=0.,
                        help='the fraction of each batch that should be acquired randomly')

    parser.add_argument('--temp-i', type=float,
                        help='the initial temperature for tempeture scaling when calculating the decay factor for cluster scaling')
    parser.add_argument('--temp-f', type=float, default=1.,
                        help='the final temperature used in the greedy metric')

    parser.add_argument('--xi', type=float, default=0.01,
                        help='the xi value to use in EI and PI metrics')
    parser.add_argument('--beta', type=int, default=2,
                        help='the beta value to use in the UCB metric')
    parser.add_argument('--threshold', type=float,
                        help='the threshold value as a positive number to use in the threshold metric')

###################################
#       OBJECTIVE ARGUMENTS       #
###################################
def add_objective_args(parser: ArgumentParser) -> None:
    parser.add_argument('-o', '--objective', 
                        choices={'lookup', 'docking'},
                        help='the objective function to use')
    parser.add_argument('--minimize', action='store_true', default=True,
                        help='whether to minimize the objective function')
    parser.add_argument('--objective-config',
                        help='the path to a configuration file containing all of the parameters with which to perform objective function evaluations')

###############################
#       MODEL ARGUMENTS       #
###############################
def add_model_args(parser: ArgumentParser) -> None:
    parser.add_argument('--model', choices=('rf', 'gp', 'nn', 'mpn'),
                        default='rf',
                        help='the model type to use')
    parser.add_argument('--test-batch-size', type=int,
                        help='the size of batch of predictions during model inference. NOTE: This has nothing to do with model training/performance and might only affect the timing of the inference step. It is only useful to play with this parameter if performance is absolutely critical.')
    parser.add_argument('--retrain-from-scratch',
                        action='store_true', default=False,
                        help='whether the model should be retrained from scratch at each iteration as opposed to retraining online.')
    parser.add_argument('--model-seed', type=int,
                        help='the random seed to use for model initialization. Not specifying will result in random model initializations each time the model is trained.')
    
    # RF args
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='the number of trees in the forest')
    parser.add_argument('--max-depth', nargs='?', type=int,
                        const=None, default=8,
                        help='the maximum depth of the tree. Not specifying this argument at all will default to 8. Adding the flag without specifying number a number will default to an unlimited depth')
    parser.add_argument('--min-samples-leaf', type=int, default=1,
                        help='the minimum number of samples required to be at a leaf node')
    # GP args
    parser.add_argument('--gp-kernel', choices={'dotproduct'},
                        default='dotproduct',
                        help='Kernel to use for Gaussian Process model')

    # MPNN args
    parser.add_argument('--init-lr', type=float, default=1e-4,
                        help='the initial learning rate for the MPNN model')
    parser.add_argument('--max-lr', type=float, default=1e-3,
                        help='the maximum learning rate for the MPNN model')
    parser.add_argument('--final-lr', type=float, default=1e-4,
                        help='the final learning rate for the MPNN model')

    # NN/MPNN args
    parser.add_argument('--conf-method', default='none',
                        choices={'ensemble', 'twooutput', 
                                 'mve', 'dropout', 'none'},
                        help='Confidence estimation method for NN/MPNN models')

    parser.add_argument('--ddp', action='store_true', default=False,
                        help='Whether to perform distributed MPN training over a multi-GPU setup via PyTorch DDP. Currently only works with CUDA >= 11.0')
    parser.add_argument('--precision', type=int, default=32, choices=(16, 32),
                        help='the precision to use when training PyTorch models in number of bits. Native precision is 32, but 16-bit precision can lead to lower memory footprint during training and faster training times on Volta GPUs. DO NOT use 16-bit precision on non-Volta GPUs. Currently only supported for single-GPU training (i.e., ddp=False)')
                        
##################################
#       STOPPING ARGUMENTS       #
##################################
def add_stopping_args(parser: ArgumentParser) -> None:
    parser.add_argument('-k', '--top-k', dest='k',
                        type=restricted_float_or_int, default=0.0005,
                        help='the top k ligands from which to calculate an average score expressed either as an integer or as a fraction of the pool')
    parser.add_argument('-w', '--window-size', type=int, default=3,
                        help='the window size to use for calculation of the moving average of the top-k scores')
    parser.add_argument('--delta', type=restricted_float, default=0.01,
                        help='the minimum acceptable difference between the moving average of the top-k scores and the current average the top-k score in order to continue exploration')
    parser.add_argument('--max-iters', type=int, default=10,
                        help='the maximum number of iterations to explore for')
    parser.add_argument('--budget', 
                        type=restricted_float_or_int, default=1.0,
                        help='the maximum budget expressed as the number of allowed objective evaluations')

def cleanup_args(args: Namespace):
    """Remove unnecessary attributes and change some arguments"""
    if isinstance(args.scores_csvs, list) and len(args.scores_csvs)==1:
        args.scores_csvs = args.scores_csvs[0]

    args.title_line = not args.no_title_line

    args_to_remove = {'no_title_line'}

    if args.metric != 'ei' or args.metric != 'pi':
        args_to_remove.add('xi')
    if args.metric != 'ucb':
        args_to_remove.add('beta')
    if args.metric != 'threshold':
        args_to_remove.add('threshold')

    if not args.cluster:
        args_to_remove |= {'temp_i', 'temp_f'}

    if args.model != 'rf':
        args_to_remove |= {'n_estimators', 'max_depth', 'min_samples_leaf'}
    if args.model != 'gp':
        args_to_remove |= {'gp_kernel'}
    if args.model != 'nn':
        args_to_remove |= set()
    if args.model != 'mpn':
        args_to_remove |= {'init_lr', 'max_lr', 'final_lr'}
    if args.model != 'nn' and args.model != 'mpn':
        args_to_remove |= {'conf_method'}

    for arg in args_to_remove:
        delattr(args, arg)

##############################
#       TYPE FUNCTIONS       #
##############################
def restricted_float_or_int(arg: str) -> Union[float, int]:
    try:
        value = int(arg)
        if value < 0:
            raise ArgumentTypeError(f'{value} is less than 0')
    except ValueError:
        value = float(arg)
        if value < 0 or value > 1:
            raise ArgumentTypeError(f'{value} must be in [0,1]')
    
    return value

def restricted_float(arg: str) -> float:
    value = float(arg)
    if value < 0 or value > 1:
        raise ArgumentTypeError(f'{value} must be in [0,1]')
    
    return value

def optional_int(arg: str):
    pass