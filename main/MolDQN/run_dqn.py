"""Execute the Training process of Deep Q Learning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import numpy as np

from absl import app
from absl import flags
from absl import logging
# from tensorflow import gfile

from logger import Logger
from model import deep_q_networks
from model import nn_utils
from rl.q_learning import q_learning_train

from task_env.envs import MultiObjectiveRewardMolecule, OptLogPMolecule

# Define flags

flags.DEFINE_string('model_dir', './checkpoints/',
                    'The directory to save data to.')
flags.DEFINE_string('target_molecule', 'C1CCC2CCCCC2C1',
                    'The SMILES string of the target molecule.')
flags.DEFINE_string('start_molecule', None,
                    'The SMILES string of the start molecule.')
flags.DEFINE_float('similarity_weight', 0.5,
                   'The weight of the similarity score in the reward function.')
flags.DEFINE_float('target_weight', 493.60,
                   'The target molecular weight of the molecule.')
flags.DEFINE_string('hparams', None,
                    'Filename for serialized HParams.')
flags.DEFINE_boolean('multi_objective', False,
                     'Whether to run multi objective DQN.')
flags.DEFINE_float('target_sas', 2.5,
                   'The target synthetic accessibility value')
flags.DEFINE_string('loss_type', 'l2',
                    'The loss type')

FLAGS = flags.FLAGS


def run_dqn(multi_objective=False):
    """Run the training of Deep Q Network

    Argument
    ------------
        -multi_objective: Boolean.
            Whether to run a multi-objective RL process.

    """

    # Read in hyper parameters
    if FLAGS.hparams is not None:
        with gfile.Open(FLAGS.hparams, 'r') as f:
            hparams = deep_q_networks.get_hparams(**json.load(f))
    else:
        hparams = deep_q_networks.get_hparams()

    # Print out hyper parameter information
    logging.info(
        'HParams:\n%s', '\n'.join([
            '\t%s: %s' % (key, value) for key, value in sorted(hparams.values().items())
        ])
    )
    logger = Logger()

    # Define environment and network
    if multi_objective:

        # In the case of multi-objective DQN
        environment = MultiObjectiveRewardMolecule(
            target_molecule=FLAGS.target_molecule,
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=False,
            allowed_ring_sizes={3, 4, 5, 6},
            max_steps=hparams.max_steps_per_episode
        )

        dqn = deep_q_networks.MultiObjectiveDeepQNetwork(
            objective_weight=np.array([[FLAGS.similarity_weight], [1 - FLAGS.similarity_weight]]),
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0
        )

    else:

        # In the case of single objective DQN
        environment = OptLogPMolecule(
            atom_types=set(hparams.atom_types),
            init_mol=FLAGS.start_molecule,
            allow_removal=hparams.allow_removal,
            allow_no_modification=hparams.allow_no_modification,
            allow_bonds_between_rings=hparams.allow_bonds_between_rings,
            allowed_ring_sizes=set(hparams.allowed_ring_sizes),
            max_steps=hparams.max_steps_per_episode
        )

        dqn = deep_q_networks.DeepQNetwork(
            input_shape=(hparams.batch_size, hparams.fingerprint_length + 1),
            q_fn=functools.partial(deep_q_networks.multi_layer_model, hparams=hparams),
            optimizer=hparams.optimizer,
            grad_clipping=hparams.grad_clipping,
            num_bootstrap_heads=hparams.num_bootstrap_heads,
            gamma=hparams.gamma,
            epsilon=1.0
        )

    # Train the DQN
    q_learning_train(
        flags=FLAGS,
        logger=logger,
        hparams=hparams,
        environment=environment,
        dqn=dqn
    )

    # Record the hyper parameters
    nn_utils.write_hparams(hparams, os.path.join(FLAGS.model_dir, 'config.json'))


def main(argv):
    del argv
    run_dqn(FLAGS.multi_objective)


if __name__ == '__main__':
    app.run(main)




