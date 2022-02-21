from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import numpy as np

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem

from six.moves import range
from six.moves import zip
import tensorflow as tf


class DeepQNetwork(object):

    def __init__(self,
                 input_shape,
                 q_fn,
                 learning_rate=0.001,
                 learning_rate_decay_step=10000,
                 learning_rate_decay_rate=0.8,
                 optimizer='Adam',
                 grad_clipping=None,
                 gamma=1.0,
                 epsilon=0.2,
                 double_q=True,
                 num_bootstrap_heads=10,
                 scope='dqn',
                 reuse=None):
        """Initialization of the Q Network

        Argument
        ------------

            - input_shape. Tuple.
                The shape of input.

            - q_fn. A function.
                The Q value function. Take observation feature as input and
                output is the Q value of the observation.

            - learning_rate. Float.
                The learning rate of optimizer.

            - learning_rate_decay_step. Int.
                The number of steps between each learning rate decay.

            - learning_rate_decay_rate. Float.
                The rate of learning rate decay.

            - optimizer. String.
                which optimizer to use.

            - grad_clipping. Boolean.
                Whether to clip gradient.

            - gamma. Float.
                Discount factor.

            - epsilon. Float.
                The probability of choosing a random action.

            - double_q. Boolean.
                Whether or not to use double Q learning.

            - num_bootstrap_head. Int.
                The number of bootstrap head to use.

            - scope. Variable scope.

            - reuse. Boolean.
                Whether or not the variable should be reused.


        """
        self.input_shape = input_shape
        self.q_fn = q_fn
        self.learning_rate = learning_rate
        self.learning_rate_decay_steps = learning_rate_decay_step
        self.learning_rate_decay_rate = learning_rate_decay_rate
        self.optimizer = optimizer
        self.grad_clipping = grad_clipping
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_bootstrap_heads = num_bootstrap_heads
        self.double_q = double_q
        self.scope = scope
        self.reuse = reuse

    def build(self):
        """Build the computational graph and training operation"""
        self._build_graph()
        self._build_training_ops()
        self._build_summary_ops()

    def _build_single_q_network(self,
                                observations,
                                head,
                                state_t,
                                state_tp1,
                                done_mask,
                                reward_t,
                                error_weight):
        """Build computational graph for single Q network

        Arguments:

            - observation. shape = [batch_size, fingerprint_length]
                The input of Q network.

            - head. shape = [1]
                The index of head chosen for decision in bootstrap DQN.

        """
        with tf.variable_scope('q_fn'):
            q_values = tf.gather(self.q_fn(observations), head, axis=-1)

        with tf.variable_scope('q_fn', reuse=True):
            q_t = self.q_fn(state_t, reuse=True)
        q_fn_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_fn')

        with tf.variable_scope('q_tp1', reuse=tf.AUTO_REUSE):
            q_tp1 = [self.q_fn(s_tp1, reuse=tf.AUTO_REUSE) for s_tp1 in state_tp1]
        q_tp1_vars = tf.trainable_variables(scope=tf.get_variable_scope().name + '/q_tp1')

        if self.double_q:

            with tf.variable_scope('q_fn', reuse=True):
                q_tp1_online = [self.q_fn(s_tp1, reuse=True) for s_tp1 in state_tp1]

            if self.num_bootstrap_heads:
                num_heads = self.num_bootstrap_heads
            else:
                num_heads = 1

            q_tp1_online_idx = [
                tf.stack([tf.argmax(q, axis=0), tf.range(num_heads, dtype=tf.int64)], axis=1) for q in q_tp1_online
            ]

            v_tp1 = tf.stack([tf.gather_nd(q, idx) for q, idx in zip(q_tp1, q_tp1_online_idx)], axis=0)

        else:
            v_tp1 = tf.stack([tf.reduce_max(q) for q in q_tp1], axis=0)

        q_tp1_masked = (1.0 - done_mask) * v_tp1
        q_t_target = reward_t + self.gamma * q_tp1_masked
        td_error = q_t - tf.stop_gradient(q_t_target)

        if self.num_bootstrap_heads:
            head_mask = tf.keras.backend.random_binomial(shape=(1, self.num_bootstrap_heads), p=0.6)
            td_error = tf.reduce_mean(td_error * head_mask, axis=1)

        errors = tf.where(
            tf.abs(td_error) < 1.0, tf.square(td_error) * 0.5,
            1.0 * (tf.abs(td_error) - 0.5)
        )
        weighted_error = tf.reduce_mean(error_weight * errors)

        return q_values, td_error, weighted_error, q_fn_vars, q_tp1_vars

    def _build_input_placeholder(self):
        """Creates the input place holder"""

        batch_size, fingerprint_length = self.input_shape

        with tf.variable_scope(self.scope, reuse=self.reuse):
            self.observations = tf.placeholder(
                tf.float32, [None, fingerprint_length], name='observation'
            )
            self.head = tf.placeholder(tf.int32, [], name='head')

            self.state_t = tf.placeholder(tf.float32, self.input_shape, name='state_t')
            self.state_tp1 = [
                tf.placeholder(tf.float32, [None, fingerprint_length], name='state_tp1_%i' % i)
                for i in range(batch_size)
            ]

            self.done_mask = tf.placeholder(tf.float32, (batch_size, 1), name='done_mask')
            self.error_weight = tf.placeholder(tf.float32, (batch_size, 1), name='error_weight')

    def _build_graph(self):
        """Build the computational graph"""
        batch_size, _ = self.input_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):
            self._build_input_placeholder()
            self.reward_t = tf.placeholder(tf.float32, (batch_size, 1), name='reward_t')
            self.q_values, self.td_error, self.weighted_error, self.q_fn_vars, self.q_tp1_vars = \
                self._build_single_q_network(self.observations,
                                             self.head,
                                             self.state_t,
                                             self.state_tp1,
                                             self.done_mask,
                                             self.reward_t,
                                             self.error_weight)
            self.action = tf.argmax(self.q_values)

    def _build_training_ops(self):
        """Create the training operations"""
        with tf.variable_scope(self.scope, reuse=self.reuse):

            self.optimization_op = tf.contrib.layers.optimize_loss(
                loss=self.weighted_error,
                global_step=tf.train.get_or_create_global_step(),
                learning_rate=self.learning_rate,
                optimizer=self.optimizer,
                clip_gradients=self.grad_clipping,
                learning_rate_decay_fn=functools.partial(
                    tf.train.exponential_decay,
                    decay_steps=self.learning_rate_decay_steps,
                    decay_rate=self.learning_rate_decay_rate
                ),
                variables=self.q_fn_vars
            )

            self.update_op = []
            for var, target in zip(sorted(self.q_fn_vars, key=lambda v: v.name),
                                   sorted(self.q_tp1_vars, key=lambda v: v.name)):
                self.update_op.append(target.assign(var))

            self.update_op = tf.group(*self.update_op)

    def _build_summary_ops(self):
        """Creates the summary operations"""
        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                self.error_summary = tf.summary.scalar('td_error', tf.reduce_mean(tf.abs(self.td_error)))
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.reward = tf.placeholder(tf.float32, [], 'summary_reward')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summary = tf.summary.scalar('reward', self.reward)
                self.episode_summary = tf.summary.merge([smiles_summary, reward_summary])

    def log_result(self, smiles, reward):
        """Summarize the SMILES and reward at the end of teh episode"""
        return tf.get_default_session().run(
            self.episode_summary,
            feed_dict={
                self.smiles: smiles,
                self.reward: reward
            }
        )

    def _run_action_op(self, observations, head):
        """Function that runs the op calculating an action given the observation"""
        return np.asscalar(tf.get_default_session().run(
            self.action,
            feed_dict={
                self.observations: observations,
                self.head: head
            }
        ))

    def get_action(self, observations, stochastic=True, head=0, update_epsilon=None):
        """Funstion that choose an action given the observations

        Argument
        ------------

            - observations. np.array. shape = [num_actions, fingerprint_length].
                The next states.

            - stochastic. Boolean.
                If set to True, all the actions are always deterministic.

            - head. Int.
                The number of bootstrap heads.

            - update_epsilon. Float or None.
                Update the epsilon to a new value.

        Return

            - action.

        """
        if update_epsilon is not None:
            self.epsilon = update_epsilon

        if stochastic and np.random.uniform() < self.epsilon:
            return np.random.randint(0, observations.shape[0])
        else:
            return self._run_action_op(observations, head)

    def train(self, states, rewards, next_states, done, weight, summary=True):
        """Function that takes a transition (s, a, r, s') and optimizes TD error

        Argument
        ------------

            - states. A batch of observations.

            - rewards. np.array, immediate reward attained after executing those actions.

            - next_states. observations that followed states.

            - done. np.array, 1 if obs_t was the last observation in the episode and
                0 otherwise obs_tp1 gets ignored, but must be of the valid shape.

            - weight. np.array, importance sampling weights for every element of the batch

            - summary. Boolean, whether to get summary.

        Return

            - TD_error. np.array. a list of differences between Q(s,a) and
                the TD target in Bellman's equation.

            - Error summary

        """
        if summary:
            ops = [self.td_error, self.error_summary, self.optimization_op]
        else:
            ops = [self.td_error, self.optimization_op]

        feed_dict= {
            self.state_t: states,
            self.reward_t: rewards,
            self.done_mask: done,
            self.error_weight: weight
        }

        for i, next_state in enumerate(next_states):
            feed_dict[self.state_tp1[i]] = next_state

        return tf.get_default_session().run(ops, feed_dict=feed_dict)


def multi_layer_model(inputs, hparams, reuse=None):
    """The network for Q value learning"""
    output = inputs

    for i, units in enumerate(hparams.dense_layers):
        output = tf.layers.dense(output, units, name='dense_%i' % i, reuse=reuse)
        output = getattr(tf.nn, hparams.activation)(output)
        if hparams.batch_norm:
            output = tf.layers.batch_normalization(output, fused=True, name='bn_%i' % i, reuse=reuse)

    if hparams.num_bootstrap_heads:
        output_dim = hparams.num_bootstrap_heads
    else:
        output_dim = 1

    output = tf.layers.dense(output, output_dim, name='final', reuse=reuse)

    return output


def get_hparams(**kwargs):
    hparams = tf.contrib.training.HParams(
        atom_types=['C', 'O', 'N'],
        max_steps_per_episode=40,
        allow_removal=True,
        allow_no_modification=True,
        allow_bonds_between_rings=False,
        allowed_ring_sizes=[3, 4, 5, 6],
        replay_buffer_size=1000000,
        learning_rate=1e-4,
        learning_rate_decay_steps=10000,
        learning_rate_decay_rate=0.8,
        num_episodes=5000,
        batch_size=64,
        learning_frequency=4,
        update_frequency=20,
        grad_clipping=10.0,
        gamma=0.9,
        double_q=True,
        num_bootstrap_heads=12,
        prioritized=False,
        prioritized_alpha=0.6,
        prioritized_beta=0.4,
        prioritized_epsilon=1e-6,
        fingerprint_radius=3,
        fingerprint_length=2048,
        dense_layers=[1024, 512, 128, 32],
        activation='relu',
        optimizer='Adam',
        batch_norm=False,
        save_frequency=1000,
        max_num_checkpoints=100,
        discount_factor=0.7
    )
    return hparams.override_from_dict(kwargs)


def get_fingerprint(smiles, hparams):
    """Get Morgan Fingerprint of a specific SMIELS string"""
    if smiles is None:
        return np.zeros((hparams.fingerprint_length, ))
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return np.zeros((hparams.fingerprint_length, ))
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, hparams.fingerprint_radius, hparams.fingerprint_length)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fingerprint, arr)
    return arr


def get_fingerprint_with_stpes_left(smiles, steps_left, hparams):
    fingerprint = get_fingerprint(smiles, hparams)
    return np.append(fingerprint, steps_left)


class MultiObjectiveDeepQNetwork(DeepQNetwork):

    def __init__(self, objective_weight, **kwargs):
        # Normalize the sum to 1.
        self.objective_weight = objective_weight / np.sum(objective_weight)
        self.num_objectives = objective_weight.shape[0]
        super(MultiObjectiveDeepQNetwork, self).__init__(**kwargs)

    def _build_graph(self):

        batch_size, _ = self.input_shape
        with tf.variable_scope(self.scope, reuse=self.reuse):

            self._build_input_placeholder()
            self.reward_t = tf.placeholder(tf.float32, (batch_size, self.num_objectives), name='reward_t')
            # objective_weight is the weight to scalarize the objective vector:
            # reward = sum (objective_weight_i * objective_i)
            self.objective_weight_input = tf.placeholder(tf.float32, [self.num_objectives, 1], name='objective_weight')

            # split reward for each q network
            rewards_list = tf.split(self.reward_t, self.num_objectives, axis=1)
            q_values_list = []
            self.td_error = []
            self.weighted_error = 0
            self.q_fn_vars = []
            self.q_tp1_vars = []

            # build a Q network for each objective
            for obj_idx in range(self.num_objectives):
                with tf.variable_scope('objective_%i' % obj_idx):
                    q_values, td_error, weighted_error, q_fn_vars, q_tp1_vars = \
                        self._build_single_q_network(
                            self.observations,
                            self.head,
                            self.state_t,
                            self.state_tp1,
                            self.done_mask,
                            rewards_list[obj_idx],
                            self.error_weight
                        )
                    q_values_list.append(tf.expand_dims(q_values, 1))
                    # td error is for summary only.
                    # weighted error is the optimization goal.
                    self.td_error.append(td_error)
                    self.weighted_error += weighted_error / self.num_objectives
                    self.q_fn_vars += q_fn_vars
                    self.q_tp1_vars += q_tp1_vars
            q_values = tf.concat(q_values_list, axis=1)
            # action is the one that leads to the maximum weighted reward.
            self.action = tf.argmax(tf.matmul(q_values, self.objective_weight_input), axis=0)

    def _build_summary_ops(self):

        with tf.variable_scope(self.scope, reuse=self.reuse):
            with tf.name_scope('summaries'):
                # The td_error here is the difference between q_t and q_t_target.
                # Without abs(), the summary of td_error is actually underestimated.
                error_summaries = [
                    tf.summary.scalar('td_error_%i' % i, tf.reduce_mean(tf.abs(self.td_error[i])))
                    for i in range(self.num_objectives)
                ]
                self.error_summary = tf.summary.merge(error_summaries)
                self.smiles = tf.placeholder(tf.string, [], 'summary_smiles')
                self.rewards = [
                    tf.placeholder(tf.float32, [], 'summary_reward_obj_%i' % i)
                    for i in range(self.num_objectives)
                ]
                # Weighted sum of the rewards.
                self.weighted_reward = tf.placeholder(tf.float32, [], 'summary_reward_sum')
                smiles_summary = tf.summary.text('SMILES', self.smiles)
                reward_summaries = [
                    tf.summary.scalar('reward_obj_%i' % i, self.rewards[i])
                    for i in range(self.num_objectives)
                ]
                reward_summaries.append(tf.summary.scalar('sum_reward', self.rewards[-1]))

                self.episode_summary = tf.summary.merge([smiles_summary] + reward_summaries)

    def log_result(self, smiles, reward):

        feed_dict = {
            self.smiles: smiles,
        }
        for i, reward_value in enumerate(reward):
            feed_dict[self.rewards[i]] = reward_value
        # calculated the weighted sum of the rewards.
        feed_dict[self.weighted_reward] = np.asscalar(np.array([reward]).dot(self.objective_weight))
        return tf.get_default_session().run(self.episode_summary, feed_dict=feed_dict)

    def _run_action_op(self, observations, head):

        return np.asscalar(tf.get_default_session().run(
                self.action,
                feed_dict={
                    self.observations: observations,
                    self.objective_weight_input: self.objective_weight,
                    self.head: head
                }))


if __name__ == '__main__':
    pass
