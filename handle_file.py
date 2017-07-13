import load as ld
import numpy as np
import csv


char_embedding=ld.load_char_embedding()
word_embedding=ld.load_word_embedding()

topic_des=ld.load_topic_des()
question_des=ld.load_question_des()

'''
topic_num=0
with open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/topic_num.txt","w") as f:
    for topic_des_ in topic_des.values:
        topic=(topic_des_[0])
        f.write(str(topic_num))
        f.write("\t")
        f.write(str(topic))
        f.write("\n")
        topic_num+=1
'''
num=0
with open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_topic_train_set.txt") as f:
    csvfile=open("/media/ada/软件/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/train.csv","w")
    writer=csv.writer(csvfile)
    writer.writerow(["question_id","topic_num","question_ct","question_wt","question_cd","question_wd"])
    question_topic =True
    while (question_topic):
        question_topic = f.readline().strip("\n").split("\t")
        num+=1
        question_id = question_topic[0]
        print(num,question_id)
        topic_id = question_topic[1].split(",")

        question_ct = \
            question_des.loc[question_des['question_id'] == int(question_id)][['question_name_char']].values[0][
                0].split(",")
        question_wt = \
            question_des.loc[question_des['question_id'] == int(question_id)][['question_name_word']].values[0][
                0].split(",")
        question_cd = \
            question_des.loc[question_des['question_id'] == int(question_id)][['question_des_char']].values[0][
                0].split(",")
        question_wd = \
            question_des.loc[question_des['question_id'] == int(question_id)][['question_des_word']].values[0][
                0].split(",")



        question_ct_embedding = []
        question_wt_embedding = []
        question_cd_embedding = []
        question_wd_embedding = []


        if (len(question_ct) ==0 or len(question_wt)==0 or len(question_cd)==0 or len(question_cd)==0):
            continue


        for question_ct_ in question_ct:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(char_embedding.loc[char_embedding['char'] == question_ct_]) == 0):
                temp = np.random.rand(256)
                question_ct_embedding.append(temp)
            else:
                question_ct_embedding.append(
                    char_embedding.loc[char_embedding['char'] == question_ct_][header_list].values[0])
        question_ct_embedding = np.mat(question_ct_embedding, dtype=np.float32)


        for question_wt_ in question_wt:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(word_embedding.loc[word_embedding['word'] == question_wt_]) == 0):
                temp = np.random.rand(256)
                question_wt_embedding.append(temp)
            else:
                question_wt_embedding.append(
                    word_embedding.loc[word_embedding['word'] == question_wt_][header_list].values[0])
        question_wt_embedding = np.mat(question_wt_embedding, dtype=np.float32)


        for question_cd_ in question_cd:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(char_embedding.loc[char_embedding['char'] == question_cd_]) == 0):
                temp = np.random.rand(256)
                question_cd_embedding.append(temp)
            else:
                question_cd_embedding.append(
                    char_embedding.loc[char_embedding['char'] == question_cd_][header_list].values[0])
        question_cd_embedding = np.mat(question_cd_embedding, dtype=np.float32)


        for question_wd_ in question_wd:
            header_list = []
            for i in range(1, 257, 1):
                header_list.append(str(i))
            if (len(word_embedding.loc[word_embedding['word'] == question_wd_]) == 0):
                temp = np.random.rand(256)
                question_wd_embedding.append(temp)
            else:
                question_wd_embedding.append(
                    word_embedding.loc[word_embedding['word'] == question_wd_][header_list].values[0])
        question_wd_embedding = np.mat(question_wd_embedding, dtype=np.float32)



        for topic_ in topic_id:
            index = 0
            # y_temp=np.zeros([2000])
            while (True):
                if (topic_des['topic_id'][index] == int(topic_)):
                    # y_temp[index] = 1
                    break
                index += 1
            writer.writerow([question_id,index,question_ct_embedding,question_wt_embedding,question_cd_embedding,question_wd_embedding])



    csvfile.close()




