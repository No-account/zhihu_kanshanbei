#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np

import tensorflow as tf





class Model():
    def __init__(self, args, deterministic=False):
        self.args = args

        if args.model == 'rnn':
            cell_fn = tf.nn.rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = tf.nn.rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = tf.nn.rnn_cell.BasicLSTMCell
        else:
            raise Exception('model type not supported: {}'.format(args.model))

        deterministic = tf.Variable(deterministic, name='deterministic')  # when training, set to False; when testing, set to True

        cell = cell_fn(args.rnn_size)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.float32, [args.batch_size, args.seq_length,args.rnn_size])
        # self.targets = tf.placeholder(tf.int64, [None, args.seq_length])  # seq2seq model
        self.targets = tf.placeholder(tf.int64, [args.batch_size])  # target is class label
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)


        '''
        with tf.variable_scope('embeddingLayer'):
            with tf.device('/cpu:0'):
                #W = tf.get_variable('W', [args.vocab_size, args.rnn_size])
                header_list=[]
                for  i in range(257):
                    header_list.append(str(i))
                W = tf.get_variable('W', ld.load_char_embedding()[header_list])
                embedded = tf.nn.embedding_lookup(W, self.input_data)

        # shape: (batch_size, seq_length, cell.input_size) => (seq_length, batch_size, cell.input_size)
                '''

        inputs = tf.split(self.input_data, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]


        #outputs, last_state = tf.nn.rnn(cell, inputs, self.initial_state, scope='rnnLayer')
        outputs, last_state = tf.nn.static_rnn(cell, inputs,self.initial_state, scope='rnnLayer')
        with tf.variable_scope('softmaxLayer'):
            softmax_w = tf.get_variable('w', [args.rnn_size, args.label_size])
            softmax_b = tf.get_variable('b', [args.label_size])
            logits = tf.matmul(outputs[-1], softmax_w) + softmax_b
            self.probs = tf.nn.softmax(logits)
        # self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, self.targets))  # Softmax loss

        self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits))  # Softmax loss
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(self.cost)  # Adam Optimizer

        self.correct_pred = tf.equal(tf.argmax(self.probs, 1), self.targets)
        self.correct_num = tf.reduce_sum(tf.cast(self.correct_pred, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def predict_label(self, sess, labels, text):
        x = np.array(text)
        state = self.cell.zero_state(len(text), tf.float32).eval()
        feed = {self.input_data: x, self.initial_state: state}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs, 1)
        id2labels = dict(zip(labels.values(), labels.keys()))
        labels = map(id2labels.get, results)
        return labels


    def predict_class(self, sess, text):
        x = np.array(text)
        state = self.cell.zero_state(len(text), tf.float32)
        feed = {self.input_data: x}
        probs, state = sess.run([self.probs, self.final_state], feed_dict=feed)

        results = np.argmax(probs,1)
        return results
