#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import tensorflow as tf





class Model():
    def __init__(self, args):

        self.input_data = tf.placeholder(tf.float32, [None, args.seq_length, args.rnn_size])
        # self.targets = tf.placeholder(tf.int64, [None, args.seq_length])  # seq2seq model
        self.targets = tf.placeholder(tf.int64, [None])  # target is class label
        self.lr = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32)

        if args.model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        cell = cell_fn(args.rnn_size)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell,output_keep_prob=self.dropout)
        self.cell  = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        #outputs, last_state = tf.nn.rnn(cell, inputs, self.initial_state, scope='rnnLayer')
        outputs, last_state = tf.nn.static_rnn(cell, inputs, scope='rnnLayer', dtype=tf.float32)
        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [args.rnn_size, args.label_size])
            softmax_b = tf.get_variable('b', [args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.targets))  # Softmax loss

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))  # Softmax loss
        self.final_state = last_state
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))



    def predict_class(self, sess, text):
        x = np.array(text)
        feed = {self.input_data: x,self.dropout:1.0}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)
        results=probs
        return results
