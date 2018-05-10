import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn
from tensorflow.python.ops.rnn_cell_impl import (
    GRUCell, LSTMCell, DropoutWrapper, MultiRNNCell, BasicRNNCell
)
import numpy as np


class RNNModel(object):

    required_params = ['num_classes', 'rnn_nodes', 'rnn_layers',
                       'keep_probability', 'cell']

    def __init__(self, element, config):
        self._validate_config(config)

        inputs = element[0]
        seq_lengths = element[1]

        with tf.variable_scope('rnn'):
            state = self._create_rnn(inputs, seq_lengths, config)
        with tf.variable_scope('output_proj'): #, regularizer=tf.contrib.layers.l1_regularizer(l2_scale)):
            w = tf.get_variable(
                    "w", [state.shape[1], config['num_classes']],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                )
            b = tf.get_variable(
                    "b", [config['num_classes']],
                    initializer=tf.truncated_normal_initializer(stddev=0.1)
                )
            self.output = tf.matmul(state, w) + b

        self.pred = tf.nn.softmax(self.output)

    def _create_rnn(self, inputs, seq_lengths, config):
        cell_fw = self._get_rnn_cell(config['cell'],
                                     config['rnn_nodes'],
                                     config['rnn_layers'],
                                     config['keep_probability'])
        cell_bw = self._get_rnn_cell(config['cell'],
                                     config['rnn_nodes'],
                                     config['rnn_layers'],
                                     config['keep_probability'])
        _, state = bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                             seq_lengths, dtype=tf.float32)
        if config['rnn_layers'] > 1:
            state = tf.concat((state[0][-1], state[1][-1]), 1)
        else:
            state = tf.concat((state[0],state[1]),1)
        state = tf.reshape(state, [-1, 2 * config['rnn_nodes']])
        return state

    def _get_rnn_cell(self, cell, RNN_nodes, RNN_layers, keep_probability):
        if cell == "GRU":
            def cell_fn():
                return GRUCell(RNN_nodes)
        if cell == "LSTM":
            def cell_fn():
                return LSTMCell(RNN_nodes)
        if cell == "RNN":
            def cell_fn():
                return RNNCell(RNN_nodes)
        if keep_probability is not None:
            def layer_fn():
                return DropoutWrapper(cell_fn(),
                                      output_keep_prob=keep_probability)
        else:
            def layer_fn():
                return cell_fn()
        if RNN_layers > 1:
            return MultiRNNCell([layer_fn() for _ in range(RNN_layers)],
                                state_is_tuple=True)
        else:
            return layer_fn()

    def loss_fn(self, targets):
        log_loss = tf.losses.softmax_cross_entropy(
                            targets,
                            self.output
                        )
        return tf.reduce_mean(log_loss)

    # def reg_loss(self, targets):
    #     regs = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #     regs = filter(lambda x: 'bias' not in x.name, regs)
    #     return self.loss_fn(targets) + tf.reduce_sum(regs)

    def acc(self, targets):
        return tf.reduce_mean(
            tf.cast(
                tf.equal(tf.argmax(self.pred, 1),
                         tf.argmax(targets, 1)),
                tf.float32)
            )

    def stream_acc(self, targets):
        acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(targets, 1), predictions=tf.argmax(self.pred,1), name="acc_metric")
        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="acc_metric")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)
        return acc_op, running_vars_initializer
        

    def _validate_config(self, config):
        for x in self.required_params:
            if x not in config:
                raise "Invalid config for RNNModel: {} required".format(x)
