import math 
import jieba
import jieba.posseg as psg 
from gensim import corpora, models
from jieba import analyse
import functools
import numpy as np 

def get_stop_word_list():
    stop_word_path = './data/stopword.txt'
    stopword_list = [sw.replace('\n', '') for sw in open(stop_word_path,encoding='utf-8').readlines()]
    return stopword_list


def seg_to_list(sentence, pos=False):
    if not pos:
        seg_list = jieba.cut(sentence)
    else:
        seg_list = psg.cut(sentence)
    return seg_list


def word_filter(seg_list, pos=False):
    stop_word_list = get_stop_word_list()
    filter_list = []
    for seg in seg_list:
        if not pos:
            word = seg 
            flag = 'n'
        else:
            word = seg.word
            flag = seg.flag 
        if not flag.startswith('n'):
            continue
        if not word in stop_word_list and len(word) > 1:
            filter_list.append(word)
    return filter_list


def load_data(pos=False, corpus_path='./data/corpus.txt'):
    doc_list = []
    for line in open(corpus_path, 'r'):
        content = line.strip()
        seg_list = seg_to_list(content, pos)
        filter_list = word_filter(seg_list, pos)
        doc_list.append(filter_list)
    return doc_list


def train_idf(doc_list):
    idf_dict = {}
    for doc in doc_list:
        for word in set(doc):
            idf_dict[word] = idf_dict.get(word, 0) + 1.0
    for k, v in idf_dict.items():
        idf_dict[k] = math.log(len(doc_list) / (1.0+v))
    default_idf = math.log(len(doc_list) / 1.0)
    return idf_dict, default_idf


def cmp(e1, e2):
    res = np.sign(e1[1] - e2[1])
    if res != 0:
        return res 
    else:
        a = e1[0] + e2[0]
        b = e2[0] + e1[0]
        if a > b:
            return 1 
        elif a == b:
            return 0
        else:
            return -1

class TfIdf(object):
    def __init__(self, idf_dict, default_dict, word_list, keyword_num):
        self.word_list = word_list
        self.idf_dict = idf_dict
        self.default_dict = default_dict
        self.keyword_num = keyword_num
        self.tf_dict = self.get_tf_dic()
    
    def get_tf_dic(self):
        tf_dict = {}
        for word in self.word_list:
            tf_dict[word] = tf_dict.get(word, 0.0) + 1.0
        tt_count = len(self.word_list)
        for k, v in tf_dict.items():
            tf_dict[k] = float(v) / tt_count
        return tf_dict

    def get_tfidf(self):
        tfidf_dict = {}
        for word in self.word_list:
            idf = self.idf_dict.get(word, self.default_dict)
            tf = self.tf_dict.get(word, 0)
            tfidf = tf * idf 
            tfidf_dict[word] = tfidf

        for k, v in sorted(tfidf_dict.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + '/ ', end='')
        print()


class TopicModel(object):
    def __init__(self, doc_list, keyword_num, model='LSI', num_topic=4):
        self.dictionary = corpora.Dictionary(doc_list)
        corpus = [self.dictionary.doc2bow(doc) for doc in doc_list]
        self.tfidf_model = models.TfidfModel(corpus)
        self.corpus_tfidf = self.tfidf_model[corpus]
        self.keyword_num = keyword_num
        self.num_topic = num_topic
        if model == 'LSI':
            self.model = self.train_lsi()
        else:
            self.model = self.train_lda()
        word_dic = self.word_dictionary(doc_list)
        self.wordtopic_dic = self.get_wordtopic(word_dic)

    def train_lsi(self):
        lsi = models.LsiModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topic)
        return lsi 
    
    def train_lda(self):
        lda = models.LdaModel(self.corpus_tfidf, id2word=self.dictionary, num_topics=self.num_topic)
        return lda

    def get_wordtopic(self, word_dict):
        wordtopic_dic = {}
        for word in word_dict:
            single_list = [word]
            wordcorpus = self.tfidf_model[self.dictionary.doc2bow(single_list)]
            wordtopic = self.model[wordcorpus]
            wordtopic_dic[word] = wordtopic
        return wordtopic_dic

        # 计算词的分布和文档的分布的相似度，取相似度最高的keyword_num个词作为关键词
    def get_simword(self, word_list):
        sentcorpus = self.tfidf_model[self.dictionary.doc2bow(word_list)]
        senttopic = self.model[sentcorpus]

        # 余弦相似度计算
        def calsim(l1, l2):
            a, b, c = 0.0, 0.0, 0.0
            for t1, t2 in zip(l1, l2):
                x1 = t1[1]
                x2 = t2[1]
                a += x1 * x1
                b += x1 * x1
                c += x2 * x2
            sim = a / math.sqrt(b * c) if not (b * c) == 0.0 else 0.0
            return sim

        # 计算输入文本和每个词的主题分布相似度
        sim_dic = {}
        for k, v in self.wordtopic_dic.items():
            if k not in word_list:
                continue
            sim = calsim(v, senttopic)
            sim_dic[k] = sim

        for k, v in sorted(sim_dic.items(), key=functools.cmp_to_key(cmp), reverse=True)[:self.keyword_num]:
            print(k + "/ ", end='')
        print()

    # 词空间构建方法和向量化方法，在没有gensim接口时的一般处理方法
    def word_dictionary(self, doc_list):
        dictionary = []
        for doc in doc_list:
            dictionary.extend(doc)

        dictionary = list(set(dictionary))

        return dictionary

    def doc2bowvec(self, word_list):
        vec_list = [1 if word in word_list else 0 for word in self.dictionary]
        return vec_list

def tfidf_extract(word_list, pos=False, keyword_num=20):
    doc_list = load_data(pos)
    idf_dic, default_idf = train_idf(doc_list)
    tfidf_model = TfIdf(idf_dic, default_idf, word_list, keyword_num)
    tfidf_model.get_tfidf()


def textrank_extract(text, pos=False, keyword_num=20):
    textrank = analyse.textrank
    keywords = textrank(text, keyword_num)
    # 输出抽取出的关键词
    for keyword in keywords:
        print(keyword + "/ ", end='')
    print()


def topic_extract(word_list, model, pos=False, keyword_num=20):
    doc_list = load_data(pos)
    topic_model = TopicModel(doc_list, keyword_num, model=model)
    topic_model.get_simword(word_list)


if __name__ == "__main__":
    text = '6月19日,《2012年度“中国爱心城市”公益活动新闻发布会》在京举行。' + \
           '中华社会救助基金会理事长许嘉璐到会讲话。基金会高级顾问朱发忠,全国老龄' + \
           '办副主任朱勇,民政部社会救助司助理巡视员周萍,中华社会救助基金会副理事长耿志远,' + \
           '重庆市民政局巡视员谭明政。晋江市人大常委会主任陈健倩,以及10余个省、市、自治区民政局' + \
           '领导及四十多家媒体参加了发布会。中华社会救助基金会秘书长时正新介绍本年度“中国爱心城' + \
           '市”公益活动将以“爱心城市宣传、孤老关爱救助项目及第二届中国爱心城市大会”为主要内容,重庆市' + \
           '、呼和浩特市、长沙市、太原市、蚌埠市、南昌市、汕头市、沧州市、晋江市及遵化市将会积极参加' + \
           '这一公益活动。中国雅虎副总编张银生和凤凰网城市频道总监赵耀分别以各自媒体优势介绍了活动' + \
           '的宣传方案。会上,中华社会救助基金会与“第二届中国爱心城市大会”承办方晋江市签约,许嘉璐理' + \
           '事长接受晋江市参与“百万孤老关爱行动”向国家重点扶贫地区捐赠的价值400万元的款物。晋江市人大' + \
           '常委会主任陈健倩介绍了大会的筹备情况。'

    pos = True
    seg_list = seg_to_list(text, pos)
    filter_list = word_filter(seg_list, pos)

    print('TF-IDF模型结果：')
    tfidf_extract(filter_list)
    print('TextRank模型结果：')
    textrank_extract(text)
    print('LSI模型结果：')
    topic_extract(filter_list, 'LSI', pos)
    print('LDA模型结果：')
    topic_extract(filter_list, 'LDA', pos)





