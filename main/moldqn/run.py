import os, torch
import numpy as np 
import sys
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('.')
from main.optimizer import BaseOptimizer
from agents.agent import DQN 

class MolDQN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "moldqn"

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        agent = DQN(
            oracle=self.oracle,
            q_fn = 'mlp', 
            n_max_oracle_call=self.args.max_oracle_calls,
            args=config,
        )

        print('Print Q function architecture:')
        print(agent.q_fn)

        global_step = 0

        for episode in range(agent.num_episodes):

            epsilon = agent.exploration.value(len(self.oracle))
            # print(f"Episode: {episode}, epsilon: {epsilon}")

            _, _ = agent.env.reset()
            head = np.random.randint(agent.num_bootstrap_heads)

            for step in range(agent.max_steps_per_episode):

                state_mol, reward, done = agent._step(epsilon=epsilon, head=head)

                # Training the network
                start_train = 50 # if self.args.noisy else 50
                if (episode > min(start_train, agent.num_episodes / 10)) and (global_step % agent.learning_frequency == 0):

                    # Update learning rate
                    if (global_step % agent.learning_rate_decay_steps == 0) and (agent.lr_schedule is not None):
                        agent.lr_schedule.step()

                    # Compute td error and optimize the network
                    td_error = agent._compute_td_loss(agent.batch_size, episode)

                    # Update the target network
                    if agent.double and (episode % agent.update_frequency == 0):
                        agent.q_fn_target.load_state_dict(agent.q_fn.state_dict())

                global_step += 1
        
            # Save checkpoint
            if episode % agent.save_frequency == 0:
                model_name = 'dqn_checkpoint_' + str(episode) + '.pth'
                torch.save(agent.q_fn.state_dict(), os.path.join(agent.log_path, model_name))

            if self.finish:
                print('max oracle hit... abort!')
                break