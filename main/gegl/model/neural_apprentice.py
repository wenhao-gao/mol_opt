import os
import json

import torch
import torch.nn as nn
from torch.distributions import Categorical
from util.smiles.char_dict import SmilesCharDictionary
from util.smiles.function import smis_to_actions


class SmilesGenerator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, lstm_dropout):
        super(SmilesGenerator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.lstm_dropout = lstm_dropout

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, output_size)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, batch_first=True, num_layers=n_layers, dropout=lstm_dropout
        )
        self.init_weights()

    def init_weights(self):
        # encoder / decoder
        nn.init.xavier_uniform_(self.encoder.weight)

        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.constant_(self.decoder.bias, 0)

        # RNN
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
                # LSTM remember gate bias should be initialised to 1
                # https://github.com/pytorch/pytorch/issues/750
                r_gate = param[int(0.25 * len(param)) : int(0.5 * len(param))]
                nn.init.constant_(r_gate, 1)

    def forward(self, x, hidden):
        embeds = self.encoder(x)
        output, hidden = self.lstm(embeds, hidden)
        output = self.decoder(output)
        return output, hidden

    @classmethod
    def load(cls, load_dir):
        model_config_path = os.path.join(load_dir, "generator_config.json")
        with open(model_config_path, "r") as f:
            config = json.load(f)

        model = cls(**config)

        model_weight_path = os.path.join(load_dir, "generator_weight.pt")
        try:
            model_state_dict = torch.load(model_weight_path, map_location="cpu")
            rnn_keys = [key for key in model_state_dict if key.startswith("rnn")]
            for key in rnn_keys:
                weight = model_state_dict.pop(key)
                model_state_dict[key.replace("rnn", "lstm")] = weight

            model.load_state_dict(model_state_dict)
        except:
            print("No pretrained weight for SmilesGenerator.")

        return model

    def save(self, save_dir):
        model_config = self.config
        model_config_path = os.path.join(save_dir, "generator_config.json")
        with open(model_config_path, "w") as f:
            json.dump(model_config, f)

        model_state_dict = self.state_dict()
        model_weight_path = os.path.join(save_dir, "generator_weight.pt")
        torch.save(model_state_dict, model_weight_path)

    @property
    def config(self):
        return dict(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            n_layers=self.n_layers,
            lstm_dropout=self.lstm_dropout,
        )


class SmilesGeneratorHandler:
    def __init__(self, model, optimizer, char_dict, max_sampling_batch_size, entropy_factor=0.0):
        self.model = model
        self.optimizer = optimizer
        self.max_sampling_batch_size = max_sampling_batch_size
        self.entropy_factor = entropy_factor
        self.char_dict = char_dict
        self.max_seq_length = self.char_dict.max_smi_len + 1

    def sample(self, num_samples, device):
        action, log_prob, seq_length = self.sample_action(num_samples=num_samples, device=device)
        smiles = self.char_dict.matrix_to_smiles(action, seq_length - 1)

        return smiles, action, log_prob, seq_length

    def sample_action(self, num_samples, device):
        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples

        action = torch.LongTensor(num_samples, self.max_seq_length).to(device)
        log_prob = torch.FloatTensor(num_samples, self.max_seq_length).to(device)
        seq_length = torch.LongTensor(num_samples).to(device)

        batch_start = 0

        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size

            action_batch, log_prob_batch, seq_length_batch = self._sample_action_batch(
                batch_size, device
            )
            action[batch_start:batch_end, :] = action_batch
            log_prob[batch_start:batch_end, :] = log_prob_batch
            seq_length[batch_start:batch_end] = seq_length_batch

            batch_start += batch_size
            remaining_samples -= batch_size

        return action, log_prob, seq_length

    def train_on_batch(self, smis, device, weights=1.0):
        actions, _ = smis_to_actions(self.char_dict, smis)
        actions = torch.LongTensor(actions)
        loss = self.train_on_action_batch(actions=actions, device=device, weights=weights)
        return loss

    def train_on_action_batch(self, actions, device, weights=1.0):
        batch_size = actions.size(0)
        batch_seq_length = actions.size(1)

        actions = actions.to(device)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * batch_seq_length, -1)

        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, batch_seq_length).mean(dim=1)
        loss = -(weights * log_target_probs).mean()

        if self.entropy_factor > 0.0:
            entropy = -torch.sum(torch.exp(log_probs) * log_probs, dim=1).mean()
            loss -= self.entropy_factor * entropy

        self.model.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def get_action_log_prob(self, actions, seq_lengths, device):
        num_samples = actions.size(0)
        actions_seq_length = actions.size(1)
        log_probs = torch.FloatTensor(num_samples, actions_seq_length).to(device)

        number_batches = (
            num_samples + self.max_sampling_batch_size - 1
        ) // self.max_sampling_batch_size
        remaining_samples = num_samples
        batch_start = 0
        for i in range(number_batches):
            batch_size = min(self.max_sampling_batch_size, remaining_samples)
            batch_end = batch_start + batch_size
            log_probs[batch_start:batch_end, :] = self._get_action_log_prob_batch(
                actions[batch_start:batch_end, :], seq_lengths[batch_start:batch_end], device
            )
            batch_start += batch_size
            remaining_samples -= batch_size

        return log_probs

    def save(self, save_dir):
        self.model.save(save_dir)

    def _get_action_log_prob_batch(self, actions, seq_lengths, device):
        batch_size = actions.size(0)
        actions_seq_length = actions.size(1)

        start_token_vector = self._get_start_token_vector(batch_size, device)
        input_actions = torch.cat([start_token_vector, actions[:, :-1]], dim=1)
        target_actions = actions

        input_actions = input_actions.to(device)
        target_actions = target_actions.to(device)

        output, _ = self.model(input_actions, hidden=None)
        output = output.view(batch_size * actions_seq_length, -1)
        log_probs = torch.log_softmax(output, dim=1)
        log_target_probs = log_probs.gather(dim=1, index=target_actions.reshape(-1, 1)).squeeze(
            dim=1
        )
        log_target_probs = log_target_probs.view(batch_size, self.max_seq_length)

        mask = torch.arange(actions_seq_length).expand(len(seq_lengths), actions_seq_length) > (
            seq_lengths - 1
        ).unsqueeze(1)
        log_target_probs[mask] = 0.0

        return log_target_probs

    def _sample_action_batch(self, batch_size, device):
        hidden = None
        inp = self._get_start_token_vector(batch_size, device)

        action = torch.zeros((batch_size, self.max_seq_length), dtype=torch.long).to(device)
        log_prob = torch.zeros((batch_size, self.max_seq_length), dtype=torch.float).to(device)
        seq_length = torch.zeros(batch_size, dtype=torch.long).to(device)

        ended = torch.zeros(batch_size, dtype=torch.bool).to(device)

        for t in range(self.max_seq_length):
            output, hidden = self.model(inp, hidden)

            prob = torch.softmax(output, dim=2)
            distribution = Categorical(probs=prob)
            action_t = distribution.sample()
            log_prob_t = distribution.log_prob(action_t)
            inp = action_t

            action[~ended, t] = action_t.squeeze(dim=1)[~ended]
            log_prob[~ended, t] = log_prob_t.squeeze(dim=1)[~ended]

            seq_length += (~ended).long()
            ended = ended | (action_t.squeeze(dim=1) == self.char_dict.end_idx).bool()

            if ended.all():
                break

        return action, log_prob, seq_length

    def _get_start_token_vector(self, batch_size, device):
        return torch.LongTensor(batch_size, 1).fill_(self.char_dict.begin_idx).to(device)
