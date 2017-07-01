import tensorflow as tf
import numpy as np
import load as ld


def Birnn(x,dropout,sequence_length):
    layers=3

    fw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
    lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * layers, state_is_tuple=True)


    bw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout)
    lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * layers, state_is_tuple=True)

    outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

    return outputs[-1]



char_embedding=ld.load_char_embedding()
word_embedding=ld.load_word_embedding()
topic_des=ld.load_topic_des()
question_topic=ld.load_question_topic()

with open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_train_set.txt") as f:
    question_des=f.readline().strip("\n").split("\t")
    question_id=question_des[0]

    question_ct=question_des[1].split(",")
    question_wt=question_des[2].split(",")
    question_cd=question_des[3].split(",")
    question_wd=question_des[4].split(",")

    topic_id=(question_topic.loc[question_topic['question_id']==int(question_id)][['topic_id']]).values
    topic_id=topic_id.reshape([1])
    topic_id=topic_id[0].split(",")

    for topic_ in topic_id:
        topic_ct = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_name_char']].values
        topic_wt = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_name_word']].values
        topic_cd = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_des_char']].values
        topic_wd = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_des_word']].values
