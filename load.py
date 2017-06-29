import pandas as pd

#返回char_embedding.txt的dataframe
def load_char_embedding():
    header_list=['char']
    for i in range(1,257,1):
        header_list.append(str(i))
    data=pd.read_csv("D:/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/char_embedding.txt",header=None,names=header_list,delim_whitespace=True,skipinitialspace=True,skiprows=1)
    return data

#返回word_embedding.txt的dataframe
def load_word_embedding():
    header_list=['word']
    for i in range(1,257,1):
        header_list.append(str(i))
    data=pd.read_csv("D:/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/char_embedding.txt",header=None,names=header_list,delim_whitespace=True,skipinitialspace=True,skiprows=1)
    return data

#返回topic_info.txt的dataframe
def load_topic_des():
    header_list=['topic_id','parent_topic','topic_name_char','topic_name_word','topic_des_char','topic_des_word']
    data=pd.read_csv("D:/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/topic_info.txt",delim_whitespace=True,header=None,names=header_list)
    return data

#返回question_topic_train_set.txt的dataframe
def load_topic_question():
    header_list=['question_id','topic_id']
    data=pd.read_csv("D:/BaiduNetdiskDownload/ieee_zhihu_cup/ieee_zhihu_cup/question_topic_train_set.txt",delim_whitespace=True,header=None,names=header_list)
    return data
