"""The training process of Q Learning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
import tensorflow as tf
from absl import logging

from rl import schedules
from rl import replay_buffer
from model.deep_q_networks import get_fingerprint


def q_learning_train(flags,
                     logger,
                     hparams,
                     environment,
                     dqn):
    """Runs the training process

    Argument
    -----------

        - flags. Abseil flags object.

        - hparams. Hyper Parameter object.

        - logger. The Tensorboard logger.

        - environment. Molecular environment.

        - dqn. Deep Q Network.

    """
    summary_writer = tf.summary.FileWriter(flags.model_dir)
    tf.reset_default_graph()

    with tf.Session() as sess:

        dqn.build()
        model_saver = tf.train.Saver(max_to_keep=hparams.max_num_checkpoints)
        # The schedule for the epsilon in epsilon greedy policy.
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
                beta_schedule=beta_schedule,
                logger=logger
            )

            if (episode + 1) % hparams.update_frequency == 0:
                sess.run(dqn.update_op)
            if (episode + 1) % hparams.save_frequency == 0:
                model_saver.save(sess, os.path.join(flags.model_dir, 'ckpt'), global_step=global_step)


def _episode(environment,
             dqn,
             memory,
             episode,
             global_step,
             hparams,
             summary_writer,
             exploration,
             beta_schedule,
             logger):
    """Run a single episode

    Argument
    ------------

        - environment. Molecular environment.

        - dqn. Deep Q Network.

        - memory. Replay buffer used to store observation and the rewards.

        - episode. Int.
            Episode number.

        - global_step. Int.
            The total number of steps across all episodes.

        - hparams. Hyper Parameter object.

        - summary_writer. FileWriter used for writing Summary protos.

        - exploration. Schedule used for exploration in environment.

        - beta_schedule. Schedule used for prioritized replay buffer.

        - logger. The Tensorboard logger.

    Return

        - global_step. Same as above.

    """

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

            # tensorboard plotting
            info = {
                'reward': result.reward  # Currently only support float reward
            }
            for tag, value in info.items():
                logger.scalar_summary(tag, value, episode)

        if (episode > min(50, hparams.num_episodes / 10)) and (global_step % hparams.learning_frequency == 0):
            if hparams.prioritized:
                state_t, _, reward_t, state_tp1, done_mask, weight, indices = \
                    memory.sample(hparams.batch_size, beta=beta_schedule.value(episode))
            else:
                state_t, _, reward_t, state_tp1, done_mask = memory.sample(hparams.batch_size)
                weight = np.ones([reward_t.shape[0]])
                indices = 0

            # Transfer to column vector
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


def _step(environment,
          dqn,
          memory,
          episode,
          hparams,
          exploration,
          head):
    """Runs a single step within an episode

    Argument
    ------------

        - environment. Molecular environment.

        - dqn. Deep Q Network.

        - memory. Replay buffer used to store observation and the rewards.

        - episode. Int.
            Episode number.

        - hparams. Hyper Parameter object.

        - exploration. Schedule used for exploration in environment.

        - head. Int.
            Number of head used for bootstrap.

    Return

        - result. Result of the step.

    """

    # Get State
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken

    # Get the valid action in current step
    valid_actions = list(environment.get_valid_actions())
    observations = np.vstack([
        np.append(get_fingerprint(act, hparams), steps_left)
        for act in valid_actions
    ])

    # Get Action
    action = valid_actions[dqn.get_action(observations, head=head, update_epsilon=exploration.value(episode))]
    action_t_fingerprint = np.append(get_fingerprint(action, hparams), steps_left)

    # Get Reward
    result = environment.step(action)

    # Get State
    steps_left = hparams.max_steps_per_episode - environment.num_steps_taken

    # Get Action
    action_fingerprints = np.vstack([
        np.append(get_fingerprint(act, hparams), steps_left)
        for act in environment.get_valid_actions()
    ])

    # store the fingerprint instead of action
    memory.add(
        obs_t=action_t_fingerprint,
        action=0,
        reward=result.reward,
        obs_tp1=action_fingerprints,
        done=float(result.terminated)
    )

    return result
