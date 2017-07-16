# -*- coding:utf-8 -*-
import argparse
import tensorflow as tf
import numpy as np
import load as ld
import pandas as pd




def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen_model_filename", default="data/frozen_model.pb", type=str,
                        help="Frozen model file to import")
    args = parser.parse_args()
    # 加载已经将参数固化后的图
    graph = load_graph(args.frozen_model_filename)

    # We can list operations
    # op.values() gives you a list of tensors it produces
    # op.name gives you the name
    # 输入,输出结点也是operation,所以,我们可以得到operation的名字
    for op in graph.get_operations():
        print(op.name, op.values())
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
    # 操作有:prefix/Placeholder/inputs_placeholder
    # 操作有:prefix/Accuracy/predictions
    # 为了预测,我们需要找到我们需要feed的tensor,那么就需要该tensor的名字
    # 注意prefix/Placeholder/inputs_placeholder仅仅是操作的名字,prefix/Placeholder/inputs_placeholder:0才是tensor的名字

    #x = graph.get_tensor_by_name('prefix/a:0')
    #y = graph.get_tensor_by_name('prefix/loss/loss:0')

    file = open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_eval.csv", "a")
    topic_info = pd.read_csv("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/topic_num.txt",
                             header=None, sep="\t")
    char_embedding = ld.load_char_embedding()
    # word_embedding = ld.load_word_embedding()
    with tf.Session(graph=graph) as sess:
            x1 = graph.get_tensor_by_name("prefix/Placeholder:0")
            # x2 = graph.get_tensor_by_name("prefix/x2:0")
            y1 = graph.get_tensor_by_name("prefix/softmaxLayer/Softmax:0")
            # y2 = graph.get_tensor_by_name("prefix/distance/strided_slice_3:0")
            num = 0
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            x_ = []
            question_id_sum=[]
            temp = []
            for i in range(256):
                temp.append(0)
            with open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_eval_set.txt",
                      "r") as f:
                question_des = True
                # for i in range(200):                                   #已经训练了200代
                #    question_topic=f.readline().strip("\n").split("\t")
                while (True):

                    question_des = f.readline().strip("\n").split("\t")
                    num += 1
                    if (num == 27581):
                        break
                while (question_des):
                    question_des = f.readline().strip("\n").split("\t")
                    num += 1

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

                    if (len(question_ct) > 40):
                        question_ct = question_ct[:40]

                    for question_ct_ in question_ct:
                        if (len(char_embedding.loc[char_embedding['char'] == question_ct_]) == 0):
                            temp = np.random.rand(256)
                            question_ct_embedding.append(temp)

                        else:
                            question_ct_embedding.append(
                                char_embedding.loc[char_embedding['char'] == question_ct_][header_list].values[0])

                    while (len(question_ct_embedding) < 40):
                        question_ct_embedding.append(temp)
                    question_ct_embedding = np.mat(question_ct_embedding, dtype=np.float32)

                    x_.append(question_ct_embedding)
                    question_id_sum.append(question_id)
                    if(len(x_)==2000):
                        print(num)
                        y_=sess.run(y1,feed_dict={x1:x_})

                        for probablity,question_id_sum_ in zip(y_,question_id_sum):
                            index = [-1 for i in range(5)]
                            index = np.array(index)

                            priority = [-1 for j in range(5)]
                            for m in range(2000):
                                for n in range(5):
                                    if (probablity[m] > priority[n]):
                                        for k in range(3, n - 1, -1):
                                            priority[k + 1] = priority[k]
                                            index[k + 1] = index[k]
                                        index[n] = m
                                        priority[n] = probablity[m]
                                        break


                            buffer = str(question_id_sum_)
                        # file.write(str(question_id))
                            for k__ in range(5):
                            # file.write(","+str(topic_info[1][index[k__]]))
                                buffer = buffer + ("," + str(topic_info[1][index[k__]]))
                    # file.write("\n")
                            buffer += ("\n")
                            file.write(buffer)
                        x_ = []
                        question_id_sum=[]

    file.close()






    print("finish")