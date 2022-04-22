"""
Utilities for various small tasks.
"""

import os
import matplotlib.pyplot as plt
import torch
import argparse

def change_str(name):
    """Remove spaces, commas, semicolons, periods, brackets from given string
    and replace them with an underscore."""

    changed = ''
    for i in range(len(name)):
        if name[i]=='{' or name[i]=='}' or name[i]=='.' or name[i]==':' \
                or name[i]==',' or name[i]==' ':
            changed += '_'
        elif name[i]=='\'':
            changed += ''
        else:
            changed += name[i]
    return changed

def make_dir(name):
    """Create a new directory."""

    if not os.path.exists(name):
        os.makedirs(name)


def closefig():
    """Clears and closes current instance of a plot."""
    plt.clf()
    plt.close()

class A:
    def __init__(self):
        A.device = torch.device('cpu:0') 


def use_gpu():
    """Connects training to gpu resources via args."""

    # if the system supports CUDA, utilize it for faster computation.
    # parser = argparse.ArgumentParser(description='Set device')
    # # parser.add_argument('--disable-cuda', action='store_false',
    # #                     help='Disable CUDA')
    # parser.add_argument('method', default='graph_ga')
    # args = parser.parse_args()
    # args.device = None
    # if torch.cuda.is_available():
    #     args.device = torch.device('cuda:0')
    # else:

    # args.device = torch.device('cpu:0')
    args = A() ##### i simplify it, so other parser would not cause error. cpu is enough for pasithea. 
    return args
