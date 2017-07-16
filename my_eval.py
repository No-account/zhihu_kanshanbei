#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, sys
import time
import argparse

import numpy as np
import tensorflow as tf

from model import Model

import load as ld
import pandas as pd


def main():
    parser = argparse.ArgumentParser()


    parser.add_argument('--model', type=str, default='lstm',
                        help='rnn, gru, lstm or bn-lstm, default lstm')

    parser.add_argument('--bn_level', type=int, default=1,
                        help='if model is bn-lstm, enable sequence-wise batch normalization with different level')

    parser.add_argument('--rnn_size', type=int, default=256,
                        help='size of RNN hidden state')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='number of layers in RNN')

    parser.add_argument('--batch_size', type=int, default=1,
                        help='minibatch size')

    parser.add_argument('--seq_length', type=int, default=40,
                        help='RNN sequence length')

    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='learning rate')

    parser.add_argument('--decay_rate', type=float, default=0.9,
                        help='decay rate for rmsprop')

    parser.add_argument('--init_from', type=str, default="./data/",
                        help='''continue training from saved model at this path. Path must contain files saved by previous training process:
                        'config.pkl'         : configuration;
                        'checkpoint'         : paths to model file(s) (created by tensorflow).
                                               Note: this file contains absolute paths, be careful when moving files around;
                        'model.ckpt-*'       : file(s) with model definition (created by tensorflow)''')
    parser.add_argument('--label_size',type=int,default=2000)
    args = parser.parse_args()
    train(args)


def train(args):
    file=open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_eval.csv","a")
    topic_info = pd.read_csv("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/topic_num.txt",
                             header=None, sep="\t")
    char_embedding = ld.load_char_embedding()
    args.vocab_size = char_embedding.shape[0]

    global ckpt
    if args.init_from is not None:
        ckpt = tf.train.get_checkpoint_state(args.init_from)

    model = Model(args,True)


    #word_embedding = ld.load_word_embedding()




    with tf.Session() as sess:

        init = tf.global_variables_initializer()
        sess.run(init)
        saver = tf.train.Saver()

        if args.init_from is not None:
            saver.restore(sess, ckpt.model_checkpoint_path)

        num = 0
        header_list = []
        for i in range(1, 257, 1):
            header_list.append(str(i))
        x_ = []
        temp = []
        for i in range(256):
            temp.append(0)
        with open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_eval_set.txt","r") as f:
            question_des = True
            # for i in range(200):                                         #已经训练了200代
            #    question_topic=f.readline().strip("\n").split("\t")
            while (True):

                question_des = f.readline().strip("\n").split("\t")
                num += 1
                if (num == 216467):
                    break
            while (question_des):
                question_des = f.readline().strip("\n").split("\t")
                num+=1

                question_id = question_des[0]

                question_ct = question_des[1].split(",")
                '''
                question_wt = \
                question_des.loc[question_des['question_id'] == int(question_id)][['question_name_word']].values[0][
                    0].split(",")
                question_cd = \
                question_des.loc[question_des['question_id'] == int(question_id)][['question_des_char']].values[0][
                    0].split(",")
                question_wd = \
                question_des.loc[question_des['question_id'] == int(question_id)][['question_des_word']].values[0][
                    0].split(",")
                    '''

                question_ct_embedding = []



                if(len(question_ct)>args.seq_length):
                    question_ct=question_ct[:args.seq_length]

                for question_ct_ in question_ct:
                    if (len(char_embedding.loc[char_embedding['char'] == question_ct_]) == 0):
                        temp = np.random.rand(256)
                        question_ct_embedding.append(temp)




                    else:
                        question_ct_embedding.append(
                            char_embedding.loc[char_embedding['char'] == question_ct_][header_list].values[0])

                while(len(question_ct_embedding)<args.seq_length):
                    question_ct_embedding.append(temp)
                question_ct_embedding = np.mat(question_ct_embedding, dtype=np.float32)



                x_.append(question_ct_embedding)
                index=[-1 for i in range(5)]
                index=np.array(index)

                probablity=model.predict_class(sess, x_)[0]

                priority=[-1 for j in range(5)]
                for m in range(2000):
                    for n in range(5):
                        if(probablity[m]>priority[n]):
                            for k in range(3,n-1,-1):
                                priority[k+1]=priority[k]
                                index[k+1]=index[k]
                            index[n] = m
                            priority[n] = probablity[m]
                            break




                print(num,question_id)
                buffer=str(question_id)
                #file.write(str(question_id))
                for k__ in range(5):
                    #file.write(","+str(topic_info[1][index[k__]]))
                    buffer=buffer+(","+str(topic_info[1][index[k__]]))
                #file.write("\n")
                buffer+=("\n")
                file.write(buffer)
                x_=[]
                if(num%10==0):
                    file.flush()



    file.close()
if __name__ == '__main__':
    main()

