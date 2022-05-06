import tensorflow as tf
from .mol_graph import max_nb
from .nn import *

def rcnn_wl_last(graph_inputs, batch_size, hidden_size, depth, training=True):
    '''This function performs the WLN embedding (local, no attention mechanism)'''
    input_atom, input_bond, atom_graph, bond_graph, num_nbs, node_mask = graph_inputs
    atom_features = tf.nn.relu(linearND(input_atom, hidden_size, "atom_embedding", init_bias=None))
    layers = []
    for i in range(depth):
        with tf.variable_scope("WL", reuse=(i>0)) as scope:
            fatom_nei = tf.gather_nd(atom_features, atom_graph)
            fbond_nei = tf.gather_nd(input_bond, bond_graph)
            h_nei_atom = linearND(fatom_nei, hidden_size, "nei_atom", init_bias=None)
            h_nei_bond = linearND(fbond_nei, hidden_size, "nei_bond", init_bias=None)
            h_nei = h_nei_atom * h_nei_bond
            mask_nei = tf.reshape(tf.sequence_mask(tf.reshape(num_nbs, [-1]), max_nb, dtype=tf.float32), [batch_size,-1,max_nb,1])
            f_nei = tf.reduce_sum(h_nei * mask_nei, -2)
            f_self = linearND(atom_features, hidden_size, "self_atom", init_bias=None)
            layers.append(f_nei * f_self * node_mask) # output
            l_nei = tf.concat([fatom_nei, fbond_nei], 3)
            nei_label = tf.nn.relu(linearND(l_nei, hidden_size, "label_U2"))
            nei_label = tf.reduce_sum(nei_label * mask_nei, -2) 
            new_label = tf.concat([atom_features, nei_label], 2)
            new_label = linearND(new_label, hidden_size, "label_U1")
            atom_features = tf.nn.relu(new_label) # updated atom features
    #kernels = tf.concat(1, layers)
    kernels = layers[-1] # atom FPs are the final output after "depth" convolutions
    fp = tf.reduce_sum(kernels, 1) # molecular FP is sum over atom FPs
    return kernels, fp

