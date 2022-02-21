from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import time
import numpy as np

from absl import app
from absl import flags
from absl import logging

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

from six.moves import range
import tensorflow as tf
from tensorflow import gfile

from model import deep_q_networks
from model import nn_utils

from rl import mol_env
from rl import mol_utils
from rl import schedules
from rl import replay_buffer


flags.DEFINE_string('model_dir',
                    './checkpoints/',
                    'The directory to save data to.')
flags.DEFINE_string('target_molecule', 'C1CCC2CCCCC2C1',
                    'The SMILES string of the target molecule.')
flags.DEFINE_string('start_molecule', None,
                    'The SMILES string of the start molecule.')
flags.DEFINE_float(
    'similarity_weight', 0.5,
    'The weight of the similarity score in the reward function.')
flags.DEFINE_float('target_weight', 493.60,
                   'The target molecular weight of the molecule.')
flags.DEFINE_string('hparams', None, 'Filename for serialized HParams.')
flags.DEFINE_boolean('multi_objective', False,
                     'Whether to run multi objective DQN.')

FLAGS = flags.FLAGS


class TargetWeightMolecule(mol_env.Molecule):
    """Defines the subclass of a molecule MDP with a target molecular weight."""

    def __init__(self, target_weight, **kwargs):
        super(TargetWeightMolecule, self).__init__(**kwargs)
        self.target_weight = target_weight

    def _reward(self):
        molecule = Chem.MolFromSmiles(self._state)
        if molecule is None:
            return -self.target_weight**2
        lower, upper = self.target_weight - 25, self.target_weight + 25
        mw = Descriptors.MolWt(molecule)
        if lower <= mw <= upper:
            return 1
        return -min(abs(lower - mw), abs(upper - mw))


class MultiObjectiveRewardMolecule(mol_env.Molecule):
    """Defines the subclass of generating a molecule with a specific reward.
    The reward is defined as a 1-D vector with 2 entries: similarity and QED
    reward = (similarity_score, qed_score)
    """

    def __init__(self, target_molecule, **kwargs):
        super(MultiObjectiveRewardMolecule, self).__init__(**kwargs)
        target_molecule = Chem.MolFromSmiles(target_molecule)
        self._target_mol_fingerprint = self.get_fingerprint(target_molecule)
        self._target_mol_scaffold = mol_utils.get_scaffold(target_molecule)
        self.reward_dim = 2

    def get_fingerprint(self, molecule):
        return AllChem.GetMorganFingerprint(molecule, radius=2)

    def get_similarity(self, smiles):
        structure = Chem.MolFromSmiles(smiles)
        if structure is None:
            return 0.0
        fingerprint_structure = self.get_fingerprint(structure)

        return DataStructs.TanimotoSimilarity(self._target_mol_fingerprint, fingerprint_structure)

    def _reward(self):
        # calculate similarity.
        # if the current molecule does not contain the scaffold of the target,
        # similarity is zero.
        if self._state is None:
            return 0.0, 0.0

        mol = Chem.MolFromSmiles(self._state)

        if mol is None:
            return 0.0, 0.0

        if mol_utils.contains_scaffold(mol, self._target_mol_scaffold):
            similarity_score = self.get_similarity(self._state)
        else:
            similarity_score = 0.0

        qed_value = QED.qed(mol)
        return similarity_score, qed_value


def run_training(hparams, environment, dqn):
    """Runs the training process"""
    summary_writer = tf.summary.FileWriter(FLAGS.model_dir)
    tf.reset_default_graph()
    with tf.Session() as sess:

        dqn.build()
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)
        exploration = schedules.PiecewiseSchedule(
            [(0, 1.0), (int(hparams.num_episodes / 2), 0.1),
             (hparams.num_episodes, 0.01)], outside_value=0.01
        )

        if hparams.prioritized:

            memory = replay_buffer.PrioritizedReplayBuffer(hparams.replay_buffer_size,
                                                           hparams.prioritized_alpha)
            beta_schedule = schedules.LinearSchedule(hparams.num_episodes,
                                                     initial_p=hparams.prioritized_beta, final_p=0)

        else:
            memory = replay_buffer.ReplayBuffer(hparams.replay_buffer_size)
            beta_schedule = None

        sess.run(tf.global_variables_initializer())
        sess.run(dqn.update_op)
        global_step = 0

        for episode in range(hparams.num_episodes):
            global_step = _episode(
                environment=environment,
                dqn=dqn,
                memory=memory,
                episode=episode,
                global_step=global_step,
                hparams=hparams,
                summary_writer=summary_writer,
                exploration=exploration,
                beta_schedule=beta_schedule
            )

            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(dqn.update_op)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(sess, os.path.join(FLAGS.model_dir, 'ckpt'), global_step=global_step)


def _episode(environment, dqn, memory, episode, global_step, hparams,
             summary_writer, exploration, beta_schedule):
    """Run a single episode"""

    episode_start_time = time.time()
    environment.initialize()

    if hparams.num_bootstrap_heads:
        head = np.random.randint(hparams.num_bootstrap_heads)
    else:
        head = 0

    for step in range(hparams.max_steps_per_episode):
        result = _step(
            environment=environment,
            dqn=dqn,
            memory=memory,
            episode=episode,
            hparams=hparams,
            exploration=exploration,
            head=head
        )

        if step == hparams.max_steps_per_episode - 1:
            episode_summary = dqn.log_result(result.state, result.reward)
            summary_writer.add_summary(episode_summary, global_step)
            logging.info('Episode %d/%d took %gs', episode + 1, hparams.num_episodes, time.time() - episode_start_time)
            logging.info('SMIELS: %s\n', result.state)
            logging.info('The reward is: %s', str(result.reward))

        if (episode > min(50, hparams.num_episodes / 10)) and (global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                state_t, _, reward_t, state_tp1, done_mask, weight, indices = \
                    memory.sample(hparams.batch_size, beta=beta_schedule.value(episode))
            else:
                state_t, _, reward_t, state_tp1, done_mask = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])

            if reward_t.ndim == 1:
                reward_t = np.expand_dims(reward_t, axis=1)

            td_error, error_summary, _ = dqn.train(
                states=state_t,
                rewards=reward_t,
                next_states=state_tp1,
                done=np.expand_dims(done_mask, axis=1),
                weight=np.expand_dims(weight, axis=1)
            )

            summary_writer.add_summary(error_summary, global_step)
            logging.info('Current TD error: %.4f', np.mean(np.abs(td_error)))

            if hparams.prioritized:
                memory.update_priorities(indices, np.abs(np.squeeze(td_error) + hparams.prioritized_epsilon).tolist())

        global_step += 1

    return global_step


def _step(environment, dqn, memory, episode, hparams, exploration, head):
    """Runs a single step within an episode"""

    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    valid_actions = list(environment.get_valid_actions())

    observations = np.vstack([
        np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
        for act in valid_actions
    ])
    action = valid_actions[dqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))]
    action_t_fingerprint = np.append(deep_q_networks.get_fingerprint(action, hparams), steps_left)

    result = environment.step(action)

    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken
    action_fingerprints = np.vstack([
        np.append(deep_q_networks.get_fingerprint(act, hparams), steps_left)
        for act in environment.get_valid_actions()
    ])

    memory.add(
        obs_t=action_t_fingerprint,
        action=0,
        reward=result.reward,
        obs_tp1=action_fingerprints,
        done=float(result.terminated)
    )

    return result


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
        environment = TargetWeightMolecule(
            target_weight=FLAGS.target_weight,
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
    run_training(
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







