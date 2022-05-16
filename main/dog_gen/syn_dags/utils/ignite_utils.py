"""
Taken from Pytorch Ignite -- just with timers.

https://github.com/pytorch/ignite
"""

import time

import numpy as np
import tabulate

from ignite.engine.engine import Engine
from torch.nn.utils import clip_grad_norm_


class AverageMeter(object):
    """
    Pytorch examples license.
    BSD 3-Clause License
    Copyright (c) 2017,
    All rights reserved.
    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice, this
      list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright notice,
      this list of conditions and the following disclaimer in the documentation
      and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
    AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
    IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
    DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
    FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
    DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
    SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
    OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
    """

    """Computes and stores the average and current value.
    taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py, licence above"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




class AvgMeterHolder:
    def __init__(self):
        self.time_to_get_batch = AverageMeter()
        self.time_to_forward = AverageMeter()
        self.time_to_step = AverageMeter()

    def reset(self):
        for elem in self.__dict__.values():
            if isinstance(elem, AverageMeter):
                elem.reset()

    def __str__(self):
        table = [
            ("Time to batch", self.time_to_get_batch.avg),
            ("Time to loss", self.time_to_forward.avg),
            ("Time for all update", self.time_to_step.avg)
        ]
        return str(tabulate.tabulate(table))


def create_unsupervised_trainer_timers(model, optimizer, loss_fn,
                              device,
                              prepare_batch,
                              output_transform=lambda x, loss: loss.item(),
                              max_norm=np.inf):
    """
    Factory function for creating a trainer for unsupervised models.
    Adapted from the regular one in Pytorch


    Note: `engine.state.output` for this engine is defined by `output_transform` parameter and is the loss
        of the processed batch by default.

    Returns:
        Engine: a trainer engine with unsupervised update function.
    """
    if device:
        model.to(device)

    timings = AvgMeterHolder()

    def _update(engine, batch):
        s_time = time.time()
        model.train()
        optimizer.zero_grad()
        x = prepare_batch(batch, device=device)
        timings.time_to_get_batch.update(time.time()-s_time)
        loss = loss_fn(model, x)
        timings.time_to_forward.update(time.time()-s_time)
        loss.backward()
        if max_norm != np.inf:
            clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        timings.time_to_step.update(time.time()-s_time)
        return output_transform(x, loss)

    return Engine(_update), timings


