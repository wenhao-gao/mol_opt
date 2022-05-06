from __future__ import print_function
import tensorflow as tf
from nn import linearND, linear
from mol_graph_direct_useScores import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph, bond_types
from models import *
import math, sys, random
from optparse import OptionParser
import threading
from multiprocessing import Queue, Pool
import time
from myrdkit import Chem
import os


parser = OptionParser()
parser.add_option("-t", "--test", dest="test_path")
parser.add_option("-p", "--cand", dest="cand_path", default=None)
parser.add_option("-a", "--ncand", dest="cand_size", default=500)
parser.add_option("-c", "--ncore", dest="core_size", default=10)
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("--checkpoint", dest="checkpoint", default="final")
parser.add_option("-v", "--verbose", dest="verbose", default=False)
opts,args = parser.parse_args()

hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
cutoff = int(opts.cand_size)
core_size = int(opts.core_size)
MAX_NCAND = int(opts.cand_size)

session = tf.Session()
_input_atom = tf.placeholder(tf.float32, [None, None, adim])
_input_bond = tf.placeholder(tf.float32, [None, None, bdim])
_atom_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_bond_graph = tf.placeholder(tf.int32, [None, None, max_nb, 2])
_num_nbs = tf.placeholder(tf.int32, [None, None])
_core_bias = tf.placeholder(tf.float32, [None])
_src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _core_bias]

q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32])
enqueue = q.enqueue(_src_holder)
input_atom, input_bond, atom_graph, bond_graph, num_nbs, core_bias = q.dequeue()

input_atom.set_shape([None, None, adim])
input_bond.set_shape([None, None, bdim])
atom_graph.set_shape([None, None, max_nb, 2])
bond_graph.set_shape([None, None, max_nb, 2])
num_nbs.set_shape([None, None])
core_bias.set_shape([None])
graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs)

with tf.variable_scope("mol_encoder"):
    fp_all_atoms = rcnn_wl_only(graph_inputs, hidden_size=hidden_size, depth=depth)

reactant = fp_all_atoms[0:1,:]
candidates = fp_all_atoms[1:,:]
candidates = candidates - reactant
candidates = tf.concat([reactant, candidates], 0)

with tf.variable_scope("diff_encoder"):
    reaction_fp = wl_diff_net(graph_inputs, candidates, hidden_size=hidden_size, depth=1)

reaction_fp = reaction_fp[1:]
reaction_fp = tf.nn.relu(linear(reaction_fp, hidden_size, "rex_hidden"))

score = tf.squeeze(linear(reaction_fp, 1, "score"), [1]) + core_bias # add in bias from CoreFinder
pred = tf.argmax(score, 0)

tk = tf.minimum(10, tf.shape(score)[0])
pred_topk_scores, pred_topk = tf.nn.top_k(score, tk)

tf.global_variables_initializer().run(session=session)

size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.trainable_variables())
sys.stderr.write("Model size: %dK\n" % (n/1000,))
sys.stderr.flush()

queue = Queue()
def read_data(coord):
    try:
        data = []
        data_f = open(opts.test_path, 'r')
        cand_f = open(opts.cand_path, 'r')

        for line in data_f:
            r,e = line.strip("\r\n ").split() # reactant smiles, true edits
            cand = cand_f.readline() # candidate bond changes from CoreFinder

            cand_split = cand.strip("\r\n ").split()
            cbonds = []  # list of (x, y, t, v)
            for i in range(1, len(cand_split), 2):
                x,y,t = cand_split[i].split('-')
                x,y = tuple(sorted([int(x) - 1, int(y) - 1]))

                # record candidate bond as (atom num, atom num, bond order, likelihood score)
                cbonds.append((x,y,float(t),float(cand_split[i+1])))

            data.append((r,cbonds))
        data_len = len(data)

        for it in range(data_len):
            reaction, cand_bonds = data[it]
            r = reaction.split('>')[0]
            ncore = core_size
            while True:
                src_tuple,conf = smiles2graph(r, None, cand_bonds, None, core_size=ncore, cutoff=MAX_NCAND, testing=True)
                if len(conf) <= MAX_NCAND:
                    break
                ncore -= 1
            queue.put((r,conf))
            feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
            session.run(enqueue, feed_dict=feed_map)

        queue.put((None, None))

    except Exception as e:
        sys.stderr.write(e)
        sys.stderr.flush()

    finally:
        coord.request_stop()


# Start parallel thread to work on processing the data
coord = tf.train.Coordinator()
data_thread = threading.Thread(target=read_data, args=(coord,))
data_thread.start()

# Restore at given checkpoint
saver = tf.train.Saver()
if opts.checkpoint:
    restore_path = os.path.join(opts.model_path, 'model.%s' % opts.checkpoint)
else:
    restore_path = tf.train.latest_checkpoint(opts.model_path)
saver.restore(session, restore_path)
sys.stderr.write('restored')
sys.stderr.flush()

total = 0.0
idxfunc = lambda x: x.GetIntProp('molAtomMapNumber')
try:
    while not coord.should_stop():
        total += 1
        r, conf = queue.get(timeout=30)
        if r is None: # reached end of data set
            break
        cur_pred = session.run(pred_topk)

        rmol = Chem.MolFromSmiles(r)
        rbonds = {}
        for bond in rmol.GetBonds():
            a1 = idxfunc(bond.GetBeginAtom())
            a2 = idxfunc(bond.GetEndAtom())
            t = bond_types.index(bond.GetBondType()) + 1
            a1, a2 = min(a1, a2), max(a1, a2)
            rbonds[(a1, a2)] = t

        if opts.verbose:
            for idx in cur_pred:
                # record the bond changes for this candidate
                for x, y, t, v in conf[idx]:
                    # convert ids to atom map numbers
                    x, y = x + 1, y + 1
                    # make sure this bond change is really a _change_
                    if ((x, y) not in rbonds and t > 0) or ((x, y) in rbonds and rbonds[(x, y)] != t):
                        print('%d-%d-%d' % (x, y, t), end=' ')
                print('|', end=' ')
            print('')

        # Report progress
        if total % 10 == 0:
            sys.stderr.write('seen {}\n'.format(total))
            sys.stderr.flush()

        if total % 1000 == 0:
            sys.stdout.flush()

except Exception as e:
    sys.stderr.write(e)
    sys.stderr.flush()
    coord.request_stop(e)
finally:
    coord.request_stop()
    coord.join([data_thread])
