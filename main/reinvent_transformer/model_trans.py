#!/usr/bin/env python

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Variable
from torch.nn import Transformer
import math
#### positional encoding ####
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        # pe.requires_grad = False
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class transformer(nn.Module):
    """ Implements a three layer GRU cell including an embedding layer
       and an output linear layer back to the size of the vocabulary"""
    def __init__(self, voc_size):
        super(transformer, self).__init__()
        num_layers=2
        in_dim=512
        self.embedding1 = nn.Embedding(voc_size, in_dim)
        self.pos_encoder = PositionalEncoding(in_dim)
        # self.transformer = Transformer(d_model=128, nhead=8,batch_first=True,num_encoder_layers=num_layers,
        #                                num_decoder_layers=num_layers)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=in_dim, nhead=8,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linear = nn.Linear(in_dim, voc_size)

    def forward(self, src):
        src = self.embedding1(src)
        src=self.pos_encoder(src)
        out = self.transformer(src)
        out = self.linear(torch.mean(out,dim=1))
        return out


class Transformer_():
    """Implements the Prior and Agent RNN. Needs a Vocabulary instance in
    order to determine size of the vocabulary and index of the END token"""
    def __init__(self, voc,device):
        self.transformer = transformer(voc.vocab_size)
        if torch.cuda.is_available():
            self.transformer.cuda()
        print('need to wait several minutes')
        self.voc = voc

    def likelihood(self, target):
        """
            Retrieves the likelihood of a given sequence

            Args:
                target: (batch_size * sequence_length) A batch of sequences

            Outputs:
                log_probs : (batch_size) Log likelihood for each example*
                entropy: (batch_size) The entropies for the sequences. Not
                                      currently used.
        """
        batch_size, seq_length = target.size()
        # print("seq_len:",seq_length)
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = torch.cat((start_token, target[:, :-1]), 1)

        log_probs = Variable(torch.zeros(batch_size).float())
        entropy = Variable(torch.zeros(batch_size))
        for step in range(seq_length):
            logits=self.transformer(x[:, :step+1])
            log_prob = F.log_softmax(logits)
            prob = F.softmax(logits)
            log_probs += NLLLoss(log_prob, target[:, step])
            entropy += -torch.sum((log_prob * prob), 1)
        return log_probs, entropy

    def sample(self, batch_size, max_length=140):
        """
            Sample a batch of sequences

            Args:
                batch_size : Number of sequences to sample
                max_length:  Maximum length of the sequences

            Outputs:
            seqs: (batch_size, seq_length) The sampled sequences.
            log_probs : (batch_size) Log likelihood for each sequence.
            entropy: (batch_size) The entropies for the sequences. Not
                                    currently used.
        """
        start_token = Variable(torch.zeros(batch_size, 1).long())
        start_token[:] = self.voc.vocab['GO']
        x = start_token

        sequences = x
        log_probs = Variable(torch.zeros(batch_size))
        finished = torch.zeros(batch_size).byte()
        entropy = Variable(torch.zeros(batch_size))
        if torch.cuda.is_available():
            finished = finished.cuda()
        for step in range(max_length):
            logits= self.transformer(sequences)
            # print('logits shape', logits.shape) ##### [128, 109]
            prob = F.softmax(logits, dim = 1)
            log_prob = F.log_softmax(logits, dim = 1)
            next_token = torch.multinomial(prob, num_samples=1)
            sequences=torch.cat([sequences,next_token],1)
            log_probs +=  NLLLoss(log_prob, next_token[:,0])
            entropy += -torch.sum((log_prob * prob), 1)

            next_token = Variable(next_token.data)
            EOS_sampled = (next_token[:,0] == self.voc.vocab['EOS']).data
            finished = torch.ge(finished + EOS_sampled, 1)
            if torch.prod(finished) == 1: break

        # sequences = torch.cat(sequences, 1)
        return sequences.data[:,1:], log_probs, entropy

def NLLLoss(inputs, targets):
    """
        Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

        Args:
            inputs : (batch_size, num_classes) *Log probabilities of each class*
            targets: (batch_size) *Target class index*

        Outputs:
            loss : (batch_size) *Loss for each example*
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss
