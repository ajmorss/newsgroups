import tensorflow as tf
from tensorflow.python.ops.rnn import dynamic_rnn
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
        with tf.variable_scope('output_proj'):
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
        cell = self._get_rnn_cell(config['cell'],
                                  config['rnn_nodes'],
                                  config['rnn_layers'],
                                  config['keep_probability'])
        _, state = dynamic_rnn(cell, inputs, seq_lengths,
                               dtype=tf.float32)
        if config['rnn_layers'] > 1:
            state = state[-1]
        state = tf.reshape(state, [-1, config['rnn_nodes']])
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
        with tf.variable_scope('loss_calc'):
            cross_entropy = tf.losses.softmax_cross_entropy(targets,
                                                            self.output)
            loss = tf.reduce_mean(cross_entropy)
        return loss

    def acc(self, targets):
        with tf.variable_scope('accuracy_calc'):
            correct = tf.cast(tf.equal(tf.argmax(self.pred, 1),
                                       tf.argmax(targets, 1)),
                              tf.float32)
            acc = tf.reduce_mean(correct)
        return acc

    def stream_acc(self, targets):
        with tf.variable_scope('streaming_acc'):
            acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(targets, 1),
                                              predictions=tf.argmax(self.pred,
                                                                    1),
                                              name="acc_metric")
            running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES,
                                             scope="acc_metric")
            running_vars_initializer = \
                tf.variables_initializer(var_list=running_vars)
        return acc_op, running_vars_initializer
        

    def _validate_config(self, config):
        for param in self.required_params:
            if param not in config:
                raise "Invalid config for RNNModel: {} required".format(param)
