import load as ld
from collections import Counter
import matplotlib.pyplot as plt
#'question_id',  'question_name_char', 'question_name_word', 'question_des_char', 'question_des_word'

question_des = ld.load_question_des()

question_title_char = question_des['question_name_char']
question_title_word = question_des['question_name_word']

question_des_char = question_des['question_des_char']
question_des_word = question_des['question_des_word']

plt.figure(1)                      #40
question_title_char_length = []
for temp in question_title_char:
    question_title_char_length.append(len(str(temp).split(",")))
counter_1 = Counter(question_title_char_length)
counter_1 = sorted(counter_1.items(),key=lambda item:item[0])
plt.plot(counter_1)


plt.figure(2)                       #30
question_title_word_length = []
for temp in question_title_word:
    question_title_word_length.append(len(str(temp).split(",")))
counter_2 = Counter(question_title_word_length)
counter_2 = sorted(counter_2.items(),key=lambda item:item[0])
plt.plot(counter_2)

plt.figure(3)                       #150
question_des_char_length = []
for temp in question_des_char:
    question_des_char_length.append(len(str(temp).split(",")))
counter_3 = Counter(question_des_char_length)
counter_3 = sorted(counter_3.items(),key=lambda item:item[0])
plt.plot(counter_3)

plt.figure(4)                       #100
question_des_word_length = []
for temp in question_des_word:
    question_des_word_length.append(len(str(temp).split(",")))
counter_4 = Counter(question_des_word_length)
counter_4 = sorted(counter_4.items(),key=lambda item:item[0])
plt.plot(counter_4)

plt.show()


