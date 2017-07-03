import tensorflow as tf
import numpy as np
import load as ld


def Birnn(x,dropout,sequence_length,scope,isTraining):
    layers=3

    with tf.name_scope("fw" + scope), tf.variable_scope("fw" + scope):
        fw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
        if(isTraining is True):
            lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout)
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_fw_cell] * layers, state_is_tuple=True)
        else:
            lstm_fw_cell_m = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * layers, state_is_tuple=True)


    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
        bw_cell = tf.nn.rnn_cell.BasicLSTMCell(sequence_length, forget_bias=1.0, state_is_tuple=True)
        if(isTraining is True):
            lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout)
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell([lstm_bw_cell] * layers, state_is_tuple=True)
        else:
            lstm_bw_cell_m = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * layers, state_is_tuple=True)


    with tf.name_scope("bw" + scope), tf.variable_scope("bw" + scope):
            outputs, _= tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, x, dtype=tf.float32)

    return outputs[-1]


x1=tf.placeholder(shape=[1,None,256],dtype=tf.float32)
x2=tf.placeholder(shape=[1,None,256],dtype=tf.float32)
isTraining=tf.placeholder(dtype=tf.bool)



char_embedding=ld.load_char_embedding()
word_embedding=ld.load_word_embedding()
topic_des=ld.load_topic_des().fillna("</s>")
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
        topic_ct = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_name_char']].values[0][0].split(",")
        topic_wt = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_name_word']].values[0][0].split(",")
        topic_cd = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_des_char']].values[0][0].split(",")
        topic_wd = topic_des.loc[topic_des['topic_id'] == int(topic_)][['topic_des_word']].values[0][0].split(",")


        topic_ct_embedding = []
        if(len(topic_ct)==0):
            topic_ct="</s>"
        for topic_ct_ in topic_ct:
            header_list=[]
            for  i in range(1,257,1):
                header_list.append(str(i))
            topic_ct_embedding.append(char_embedding.loc[char_embedding['char']==topic_ct_][header_list].values[0])
        topic_ct_embedding=np.mat(topic_ct_embedding,dtype=np.float32)
        topic_ct_embedding=np.resize(topic_ct_embedding,(1,-1,256))


        question_ct_embedding=[]
        if(len(question_ct)==0):
            question_ct="</s>"
        for question_ct_ in question_ct:
            header_list=[]
            for i in range(1,257,1):
                header_list.append(str(i))
            question_ct_embedding.append(char_embedding.loc[char_embedding['char']==question_ct_][header_list].values[0])
        question_ct_embedding=np.mat(question_ct_embedding,dtype=np.float32)
        question_ct_embedding=np.resize(question_ct_embedding,(1,-1,256))

        out1 = Birnn(x1, 0.5, 256, "side1", isTraining)
        out2 = Birnn(x2, 0.5, 256, "side2", isTraining)
        print(question_ct_embedding)
        print(question_ct_embedding.shape)
        print(topic_ct_embedding)
        print(topic_ct_embedding.shape)
        '''
        distance1 = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(out1, out2)), 1, keep_dims=True))
        distance2 = tf.div(distance1, tf.add(tf.sqrt(tf.reduce_sum(tf.square(out1), 1, keep_dims=True)),tf.sqrt(tf.reduce_sum(tf.square(out2), 1, keep_dims=True))))
        distance3 = tf.reshape(distance2, [-1], name="distance")
'''
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            a,b=sess.run([out1,out2],feed_dict={x1:question_ct_embedding,x2:topic_ct_embedding,isTraining:True})

            print(a.shape)
            print(" b:")
            print(b.shape)
