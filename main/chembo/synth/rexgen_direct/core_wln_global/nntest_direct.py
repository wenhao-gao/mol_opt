from __future__ import print_function
import tensorflow as tf
from .nn import linearND, linear
from .models import *
from .ioutils_direct import *
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue
import os

'''
Script for testing the core finder model and outputting predictions.

Model architecture comments can be found in the training script
'''

NK3 = 80
NK2 = 40
NK1 = 20
NK0 = 16
NK = 12

parser = OptionParser()
parser.add_option("-t", "--test", dest="train_path")
parser.add_option("-m", "--model", dest="model_path")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-r", "--rich", dest="rich_feat", default=False)
parser.add_option("-c", "--checkpoint", dest="checkpoint", default="final")
parser.add_option("-v", "--verbose", dest="verbose", default=False)
parser.add_option("--hard", dest="hard", default=False) # whether to allow reagents/solvents to contribute atoms
parser.add_option("--detailed", dest="detailed", default=False) # whether to include scores in output
# note: explicitly including scores (--detailed true) is very important for model performance!
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
detailed = bool(opts.detailed)

if opts.rich_feat:
    from .mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
else:
    from .mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g

smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

gpu_options = tf.GPUOptions()
session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
_input_atom = tf.placeholder(tf.float32, [batch_size, None, adim])
_input_bond = tf.placeholder(tf.float32, [batch_size, None, bdim])
_atom_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
_bond_graph = tf.placeholder(tf.int32, [batch_size, None, max_nb, 2])
_num_nbs = tf.placeholder(tf.int32, [batch_size, None])
_node_mask = tf.placeholder(tf.float32, [batch_size, None])
_src_holder = [_input_atom, _input_bond, _atom_graph, _bond_graph, _num_nbs, _node_mask]
_label = tf.placeholder(tf.int32, [batch_size, None])
_binary = tf.placeholder(tf.float32, [batch_size, None, None, binary_fdim])

q = tf.FIFOQueue(100, [tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.float32])
enqueue = q.enqueue(_src_holder + [_label, _binary])
input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask, label, binary = q.dequeue()

input_atom.set_shape([batch_size, None, adim])
input_bond.set_shape([batch_size, None, bdim])
atom_graph.set_shape([batch_size, None, max_nb, 2])
bond_graph.set_shape([batch_size, None, max_nb, 2])
num_nbs.set_shape([batch_size, None])
node_mask.set_shape([batch_size, None])
label.set_shape([batch_size, None])
binary.set_shape([batch_size, None, None, binary_fdim])

node_mask = tf.expand_dims(node_mask, -1)

graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)
with tf.variable_scope("encoder"):
    atom_hiddens, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

atom_hiddens1 = tf.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
atom_hiddens2 = tf.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
atom_pair = atom_hiddens1 + atom_hiddens2

att_hidden = tf.nn.relu(linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="att_bin_feature"))
att_score = linearND(att_hidden, 1, scope="att_scores")
att_score = tf.nn.sigmoid(att_score)
att_context = att_score * atom_hiddens1
att_context = tf.reduce_sum(att_context, 2)

att_context1 = tf.reshape(att_context, [batch_size, 1, -1, hidden_size])
att_context2 = tf.reshape(att_context, [batch_size, -1, 1, hidden_size])
att_pair = att_context1 + att_context2

pair_hidden = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="bin_feature", init_bias=None) + linearND(att_pair, hidden_size, scope="ctx_feature")
pair_hidden = tf.nn.relu(pair_hidden)
pair_hidden = tf.reshape(pair_hidden, [batch_size, -1, hidden_size])

score = linearND(pair_hidden, 5, scope="scores")
score = tf.reshape(score, [batch_size, -1])
bmask = tf.to_float(tf.equal(label, INVALID_BOND)) * 10000
topk_scores, topk = tf.nn.top_k(score - bmask, k=NK3)
label_dim = tf.shape(label)[1]

tf.global_variables_initializer().run(session=session)
size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.trainable_variables())
if opts.verbose:
    sys.stderr.write("Model size: %dK" % (n/1000,))
else:
    print("Model size: %dK" % (n/1000,))

# Restore
saver = tf.train.Saver()
if opts.checkpoint:
    restore_path = os.path.join(opts.model_path, 'model.%s' % opts.checkpoint)
else:
    restore_path = tf.train.latest_checkpoint(opts.model_path)
saver.restore(session, restore_path)
sys.stderr.write('restored')
sys.stderr.flush()

queue = Queue()
def read_data(path, coord):
    data = []
    with open(path, 'r') as f:
        for line in f:
            r,e = line.strip("\r\n ").split()
            data.append((r,e))
    if not opts.verbose:
        print('Data length: {}'.format(len(data)))

    for it in range(0, len(data), batch_size):
        src_batch, edit_batch = [], []; all_ratoms = []; all_rbonds = []; react_batch = []
        for i in range(batch_size):
            react,_,p = data[it][0].split('>')
            src_batch.append(react)
            edits = data[it][1]
            edit_batch.append(edits)
            react_batch.append(react)
            it = (it + 1) % len(data)

            pmol = Chem.MolFromSmiles(p)
            patoms = set([atom.GetAtomMapNum() for atom in pmol.GetAtoms()])
            mapnum = max(patoms) + 1

            # ratoms, rbonds keep track of what parts of the reactant molecules are involved in the reaction
            ratoms = []; rbonds = []
            new_mapnums = False
            react_new = []
            for x in react.split('.'):
                xmol = Chem.MolFromSmiles(x)
                xatoms = [atom.GetAtomMapNum() for atom in xmol.GetAtoms()]
                if len(set(xatoms) & patoms) > 0 or opts.hard:
                    ratoms.extend(xatoms)
                    rbonds.extend([
                        tuple(sorted([b.GetBeginAtom().GetAtomMapNum(), b.GetEndAtom().GetAtomMapNum()]) + [b.GetBondTypeAsDouble()]) \
                        for b in xmol.GetBonds()
                    ])
            all_ratoms.append(ratoms)
            all_rbonds.append(rbonds)

        # Prepare batch for TF
        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        feed_map.update({_label:cur_label, _binary:cur_bin})
        if detailed:
            queue.put((sp_label, all_ratoms, all_rbonds, edit_batch, react_batch))
        else:
            queue.put((sp_label, all_ratoms, all_rbonds, edit_batch))
        session.run(enqueue, feed_dict=feed_map)

    if detailed:
        queue.put((None, None, None, None, None))
    else:
        queue.put((None, None, None, None))

# Start data processing thread
coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(opts.train_path, coord))
t.start()

it, sum_acc, sum_err = 0, 0.0, 0.0
accNK = 0.; accNK0 = 0.; accNK1 = 0.; accNK2 = 0.; accNK3 = 0.
bo_to_index  = {0.0: 0, 1.0:1, 2.0:2, 3.0:3, 1.5:4}
bindex_to_o = {val:key for key, val in bo_to_index.iteritems()}
nbos = len(bo_to_index)
try:
    while not coord.should_stop():
        if detailed:
            (sp_label, all_ratoms, all_rbonds, edit_batch, react_batch) = queue.get(20)
        else:
            (sp_label, all_ratoms, all_rbonds, edit_batch) = queue.get(20)
        if sp_label is None:
            break # done

        if detailed:
            cur_topk, cur_sco, cur_dim = session.run([topk, topk_scores, label_dim])
        else:
            cur_topk, cur_dim = session.run([topk, label_dim])
        cur_dim = int(math.sqrt(cur_dim/5)) # important! get num atoms

        for i in range(batch_size):
            ratoms = all_ratoms[i]
            rbonds = all_rbonds[i]
            pre = 0

            # Keep track of different accuracies
            for j in range(NK):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: accNK += 1
            pre = 0
            for j in range(NK0):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: accNK0 += 1
            pre = 0
            for j in range(NK1):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: accNK1 += 1
            pre = 0
            for j in range(NK2):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: accNK2 += 1
            pre = 0
            for j in range(NK3):
                if cur_topk[i, j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: accNK3 += 1

            if opts.verbose:
                if detailed:
                    print("{}".format(react_batch[i]), end=' ')
                for j in range(NK3):
                    k = cur_topk[i,j] # index that must be converted to (x, y, t) tuple
                    bindex = k % nbos
                    y = ((k - bindex) / nbos) % cur_dim + 1
                    x = (k - bindex - (y-1) * nbos) / cur_dim / nbos + 1
                    bo = bindex_to_o[bindex]
                    # Only allow atoms from reacting molecules to be part of the prediction,
                    # for consistency with Schwaller et al. seq2seq evaluation
                    if x < y and x in ratoms and y in ratoms and (x, y, bo) not in rbonds:
                        print("{}-{}-{:.1f}".format(x, y, bo), end=' ')
                        if detailed: # include actual score of prediction
                            print("{:.3f}".format(cur_sco[i, j]), end=' ')
                print('') # new line

        it += 1
        if it % 5 == 0:
            tot_samples = batch_size * it
            sys.stderr.write('After seeing {}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}\n'.format(
                tot_samples, NK, accNK/tot_samples, NK0, accNK0/tot_samples, NK1, accNK1/tot_samples, NK2, accNK2/tot_samples, NK3, accNK3/tot_samples,
            ))
            sys.stderr.flush()

except Exception as e:
    sys.stderr.write(e)
    if not opts.verbose:
        print(e)
    coord.request_stop(e)
finally:
    tot_samples = batch_size * it
    if tot_samples > 0:
        sys.stderr.write(
            'After seeing {}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}, acc@{}: {:.3f}\n'.format(
                tot_samples, NK, accNK/tot_samples, NK0, accNK0/tot_samples, NK1, accNK1/tot_samples, NK2, accNK2/tot_samples, NK3, accNK3/tot_samples,
        ))
        sys.stderr.flush()
    coord.request_stop()
    coord.join([t])
