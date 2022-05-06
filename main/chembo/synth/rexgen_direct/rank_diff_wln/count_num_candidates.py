import tensorflow as tf
from nn import linearND, linear
from mol_graph_direct import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, smiles2graph, bond_types
from models import *
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue
from myrdkit import rdkit
from myrdkit import Chem
import os


'''
This function uses the smiles2graph function from mol_graph_direct to keep track of how many
valid candidates are generated when restricting enumeration to use a certain number of candidate bonds
and core size.
'''

parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-p", "--cand", dest="cand_path", default=None)
parser.add_option("-a", "--ncand", dest="cand_size", default=500)
parser.add_option("-c", "--ncore", dest="core_size", default=10)
opts,args = parser.parse_args()

core_size = int(opts.core_size)
MAX_NCAND = int(opts.cand_size)

data = []
data_f = open(opts.test_path, 'r')
cand_f = open(opts.cand_path, 'r')
num_cands = open(opts.cand_path + '.num_cands_core{}.txt'.format(core_size), 'w')

for line in data_f:
    r,e = line.strip("\r\n ").split()
    cand = cand_f.readline()
    cbonds = []

    for b in cand.strip("\r\n ").split():
        x,y,t = b.split('-')
        x,y,t = int(x)-1,int(y)-1,float(t)
        cbonds.append((x,y,t))

    data.append((r,cbonds))

data_len = len(data)
for it in range(data_len):
    reaction, cand_bonds = data[it]
    r = reaction.split('>')[0]
    ncore = core_size
    src_tuple,conf = smiles2graph(r, None, cand_bonds, None, core_size=ncore, cutoff=10000000, testing=True)
    num_cands.write('{}\n'.format(len(conf)))
    if it % 100 == 0:
        print('Done {}'.format(it))

data_f.close()
cand_f.close()
num_cands.close()
