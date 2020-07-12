import os 

data_file = '/home/peihongyue/data/code/learning-nlp/chapter-8/sentiment-analysis/'

pos_files = [data_file + 'pos/' + f for f in os.listdir(data_file + 'pos/') if os.path.isfile(data_file + 'pos/' + f)]
neg_files = [data_file + 'neg/' + f for f in os.listdir(data_file + 'neg/') if os.path.isfile(data_file + 'neg/' + f)]


num_words = []

for f in pos_files:
    with open(f) as f:
        line = f.readline()
        num_words.append(len(line.split()))

for f in neg_files:
    with open(f) as f:
        line = f.readline()
        num_words.append(len(line.split()))


import matplotlib.pyplot as plt 

plt.hist(num_words, 50)
plt.show()





