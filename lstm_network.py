import tensorflow as tf
import numpy as np
import load as ld

class simase_lstm:
    def Birnn(self,x,dropout,embedding_size,sequence_length):
        layers=3

        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
        lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
        lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * layers, state_is_tuple=True)


        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
        lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout)
        lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * layers, state_is_tuple=True)

        outputs, _, _ = tf.nn.bidirectional_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

        return outputs[-1]

sl=simase_lstm()
char_embedding=ld.load_char_embedding()
word_embedding=ld.load_word_embedding()
topic_des=ld.load_topic_des()
question_topic=ld.load_question_topic()

with open("./ieee_zhihu_cup/question_train_set.txt") as f:
    question_des=f.readline().strip("\n").split("\t")
    question_id=question_des[0]
    question_ct=question_des[1].split(",")
    question_wt=question_des[2].split(",")
    question_cd=question_des[3].split(",")
    question_wd=question_des[4].split(",")

    print(question_id)
    topic_id=str(question_topic.loc[question_topic['question_id']==int(question_id)]['topic_id']).split(",")

    print(topic_id[0])




