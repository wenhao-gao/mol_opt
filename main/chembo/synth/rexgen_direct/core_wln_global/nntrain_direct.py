from __future__ import print_function
import tensorflow as tf
from .nn import linearND, linear
from .mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
from .models import *
from .ioutils_direct import *
import math, sys, random
from collections import Counter
from optparse import OptionParser
from functools import partial
import threading
from multiprocessing import Queue

'''
Script for training the core finder model

Key changes from NIPS paper version:
- Addition of "rich" options for atom featurization with more informative descriptors
- Predicted reactivities are not 1D, but 5D and explicitly identify what the bond order of the product should be
'''

NK = 20
NK0 = 10

parser = OptionParser()
parser.add_option("-t", "--train", dest="train_path")
parser.add_option("-m", "--save_dir", dest="save_path")
parser.add_option("-b", "--batch", dest="batch_size", default=20)
parser.add_option("-w", "--hidden", dest="hidden_size", default=100)
parser.add_option("-d", "--depth", dest="depth", default=1)
parser.add_option("-l", "--max_norm", dest="max_norm", default=5.0)
parser.add_option("-r", "--rich", dest="rich_feat", default=False)
opts,args = parser.parse_args()

batch_size = int(opts.batch_size)
hidden_size = int(opts.hidden_size)
depth = int(opts.depth)
max_norm = float(opts.max_norm)
if opts.rich_feat:
    from .mol_graph_rich import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g
else:
    from .mol_graph import atom_fdim as adim, bond_fdim as bdim, max_nb, smiles2graph_list as _s2g

smiles2graph_batch = partial(_s2g, idxfunc=lambda x:x.GetIntProp('molAtomMapNumber') - 1)

gpu_options = tf.GPUOptions(allow_growth=True)
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

# Queueing system allows CPU to prepare a buffer of <100 batches
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
flat_label = tf.reshape(label, [-1])
bond_mask = tf.to_float(tf.not_equal(flat_label, INVALID_BOND))
flat_label = tf.maximum(0, flat_label)

# Perform the WLN embedding 
graph_inputs = (input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask)
with tf.variable_scope("encoder"):
    atom_hiddens, _ = rcnn_wl_last(graph_inputs, batch_size=batch_size, hidden_size=hidden_size, depth=depth)

# Calculate local atom pair features as sum of local atom features
atom_hiddens1 = tf.reshape(atom_hiddens, [batch_size, 1, -1, hidden_size])
atom_hiddens2 = tf.reshape(atom_hiddens, [batch_size, -1, 1, hidden_size])
atom_pair = atom_hiddens1 + atom_hiddens2

# Calculate attention scores for each pair o atoms
att_hidden = tf.nn.relu(linearND(atom_pair, hidden_size, scope="att_atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="att_bin_feature"))
att_score = linearND(att_hidden, 1, scope="att_scores")
att_score = tf.nn.sigmoid(att_score)

# Calculate context features using those attention scores
att_context = att_score * atom_hiddens1
att_context = tf.reduce_sum(att_context, 2)

# Calculate global atom pair features as sum of atom context features
att_context1 = tf.reshape(att_context, [batch_size, 1, -1, hidden_size])
att_context2 = tf.reshape(att_context, [batch_size, -1, 1, hidden_size])
att_pair = att_context1 + att_context2

# Calculate likelihood of each pair of atoms to form a particular bond order
pair_hidden = linearND(atom_pair, hidden_size, scope="atom_feature", init_bias=None) + linearND(binary, hidden_size, scope="bin_feature", init_bias=None) + linearND(att_pair, hidden_size, scope="ctx_feature")
pair_hidden = tf.nn.relu(pair_hidden)
pair_hidden = tf.reshape(pair_hidden, [batch_size, -1, hidden_size])
score = linearND(pair_hidden, 5, scope="scores")
score = tf.reshape(score, [batch_size, -1])

# Mask existing/invalid bonds before taking topk predictions
bmask = tf.to_float(tf.equal(label, INVALID_BOND)) * 10000
_, topk = tf.nn.top_k(score - bmask, k=NK)
flat_score = tf.reshape(score, [-1])

# Train with categorical crossentropy
loss = tf.nn.sigmoid_cross_entropy_with_logits(flat_score, tf.to_float(flat_label))
loss = tf.reduce_sum(loss * bond_mask)

# Use Adam with clipped gradients
_lr = tf.placeholder(tf.float32, [])
optimizer = tf.train.AdamOptimizer(learning_rate=_lr)
param_norm = tf.global_norm(tf.trainable_variables())
grads_and_vars = optimizer.compute_gradients(loss / batch_size) #+ beta * param_norm)
grads, var = zip(*grads_and_vars)
grad_norm = tf.global_norm(grads)
new_grads, _ = tf.clip_by_global_norm(grads, max_norm)
grads_and_vars = zip(new_grads, var)
backprop = optimizer.apply_gradients(grads_and_vars)

tf.global_variables_initializer().run(session=session)
size_func = lambda v: reduce(lambda x, y: x*y, v.get_shape().as_list())
n = sum(size_func(v) for v in tf.trainable_variables())
print("Model size: %dK" % (n/1000,))

# Multiprocessing queue to run in parallel to Tensorflow queue, contains aux. information
queue = Queue()

def count(s):
    c = 0
    for i in range(len(s)):
        if s[i] == ':':
            c += 1
    return c

def read_data(path, coord):
    '''Process data from a text file; bin by number of heavy atoms
    since that will determine the input sizes in each batch'''
    bucket_size = [10,20,30,40,50,60,80,100,120,150]
    buckets = [[] for i in range(len(bucket_size))]
    with open(path, 'r') as f:
        for line in f:
            r,e = line.strip("\r\n ").split()
            c = count(r)
            for i in range(len(bucket_size)):
                if c <= bucket_size[i]:
                    buckets[i].append((r,e))
                    break

    for i in range(len(buckets)):
        random.shuffle(buckets[i])
    
    head = [0] * len(buckets)
    avil_buckets = [i for i in range(len(buckets)) if len(buckets[i]) > 0]
    while True:
        src_batch, edit_batch = [], []
        bid = random.choice(avil_buckets)
        bucket = buckets[bid]
        it = head[bid]
        data_len = len(bucket)
        for i in range(batch_size):
            react = bucket[it][0].split('>')[0]
            src_batch.append(react)
            edits = bucket[it][1]
            edit_batch.append(edits)
            it = (it + 1) % data_len
        head[bid] = it

        # Prepare batch for TF
        src_tuple = smiles2graph_batch(src_batch)
        cur_bin, cur_label, sp_label = get_all_batch(zip(src_batch, edit_batch))
        feed_map = {x:y for x,y in zip(_src_holder, src_tuple)}
        feed_map.update({_label:cur_label, _binary:cur_bin})
        session.run(enqueue, feed_dict=feed_map)
        queue.put(sp_label)

    coord.request_stop()

coord = tf.train.Coordinator()
t = threading.Thread(target=read_data, args=(opts.train_path, coord))
t.start()

saver = tf.train.Saver(max_to_keep=None)
it, sum_acc, sum_err, sum_gnorm = 0, 0.0, 0.0, 0.0
lr = 0.001
try:
    while not coord.should_stop():
        it += 1
        # Run one minibatch
        _, cur_topk, pnorm, gnorm = session.run([backprop, topk, param_norm, grad_norm], feed_dict={_lr:lr})
        sp_label = queue.get()
        # Get performance
        for i in range(batch_size):
            pre = 0
            for j in range(NK):
                if cur_topk[i,j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: sum_err += 1
            pre = 0
            for j in range(NK0):
                if cur_topk[i,j] in sp_label[i]:
                    pre += 1
            if len(sp_label[i]) == pre: sum_acc += 1
        sum_gnorm += gnorm

        if it % 50 == 0:
            print("Acc@10: %.4f, Acc@20: %.4f, Param Norm: %.2f, Grad Norm: %.2f" % (sum_acc / (50 * batch_size), sum_err / (50 * batch_size), pnorm, sum_gnorm / 50) )
            sys.stdout.flush()
            sum_acc, sum_err, sum_gnorm = 0.0, 0.0, 0.0
        if it % 10000 == 0:
            lr *= 0.9
            saver.save(session, opts.save_path + "/model.ckpt", global_step=it)
            print("Model Saved!")
except Exception as e:
    print(e)
    coord.request_stop(e)
finally:
    saver.save(session, opts.save_path + "/model.final")
    coord.request_stop()
    coord.join([t])
