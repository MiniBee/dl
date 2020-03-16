#!/usr/bin/env python
# _*_coding:utf-8 _*_
#@Time    :2020/2/28 下午2:18
#@Author  :hongyue pei 
#@FileName: data_flow.py
#@Software: PyCharm
from collections import Counter
import jieba
import numpy as np
import re
import heapq

import sif as sif
import similarity


def word2Vec(file):
    wordsVec = {}
    with open(file) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            key = line.split()[0]
            vec = [float(x) for x in line.split()[1:]]
            wordsVec[key] = vec
    return wordsVec


def getWordFrequency(tokens_list):
    all_tokens = []
    for tokens in tokens_list:
        all_tokens.extend(tokens)
    c = Counter(all_tokens)
    with open('/home/peihongyue/project/python/dl/data/autoSummary/word_frequence', 'w') as f:
        for i in c:
            f.write(i + ',' + str(c.get(i)) + '\n')


def getWordWeight(wordFrequnceFile, a):
    wordWeight = {}
    N = 0
    with open(wordFrequnceFile) as f:
        lines = f.readlines()
    for line in lines:
        word = line.split(',')[0]
        count = float(line.split(',')[1])
        wordWeight[word] = count
        N += count
    for word, count in wordWeight.items():
        wordWeight[word] = a / (a + (count/N))
    return wordWeight


def sentences2idx(sentences, wordVec):
    n_samples = len(sentences)
    lengths = [len(s) for s in sentences]
    maxlen = np.max(lengths)
    ret = []
    for i, sentence in enumerate(sentences):
        ret.append(getSentenceVec(list(jieba.cut(sentence)), wordVec))
    return np.array(ret)


def getSentenceVec(sentence, wordVec):
    ret = []
    # print(sentence)
    for i in sentence:
        ret.append([i] + getWordVec(i, wordVec))
    return np.array(ret)


def getWordVec(word, wordVec):
    if word in wordVec:
        return wordVec[word]
    else:
        # print('noword', word)
        return ['0'] * 100


def token(string, split=None):
    if not split:
        # 训练
        return re.findall('\w+', string)
    else:
        # 预测
        return re.split(split, string)


def getSentences(string):
    return [i for i in token(string, split=u'，|。') if len(i) > 0]


def getContent(sentences):
    ret = []
    for s in sentences:
        ret.extend(list(jieba.cut(s)))
    return np.array(ret)


def mostSimilar(sentencesVec, contentVec, length=0):
    if length == 0:
        length = len(sentencesVec) // 3
    # print(length)
    contentVec = contentVec.reshape(100)
    sentencesSimilarity = []
    for sentenceVec in sentencesVec:
        # print(sentenceVec.shape, contentVec.shape)
        sentencesSimilarity.append(similarity.pearsonDistance(sentenceVec, contentVec))
    sentencesSimilarity = similarity.smooth(sentencesSimilarity)
    # print(sentencesSimilarity)
    min_value_index = map(sentencesSimilarity.index, heapq.nsmallest(length, sentencesSimilarity))
    # min_value_index = similarity.smooth(min_value_index)
    return sorted(min_value_index)


def main(sentences, title):
    wordVec = word2Vec('/home/peihongyue/project/python/dl/data/autoSummary/word2vec')
    wordWeight = getWordWeight('/home/peihongyue/project/python/dl/data/autoSummary/word_frequence', 1e-7)
    # sentences = ['手机农历年发布', '第十一回国家与狼共舞']
    sentencesVec = sentences2idx(sentences, wordVec)
    content = getContent(sentences)
    contentVec = [getSentenceVec(content, wordVec)]
    title = getContent(title)

    titleVec = [getSentenceVec(title, wordVec)]
    print('---')
    # print(np.array([contentVec]).shape)
    # print(sentencesVec.shape)
    sentencesVec_sif = sif.SIF(wordWeight, sentencesVec).__call__()
    contentVec_sif = sif.SIF(wordWeight, contentVec).__call__()
    titleVec_sif = sif.SIF(wordWeight, titleVec).__call__()
    min_value_index = mostSimilar(sentencesVec_sif, contentVec_sif, 10)
    print('，'.join([sentences[i] for i in min_value_index]))
    # print(contentVec_sif.shape)
    # print(titleVec_sif.shape)
    # print(sentencesVec_sif.shape)
    print('end sif')


if __name__ == '__main__':
    # contect = '此外，自本周（6月12日）起，除小米手机6等15款机型外，其余机型已暂停更新发布（含开发版/体验版内测，稳定版暂不受影响），以确保工程师可以集中全部精力进行系统优化工作。有人猜测这也是将精力主要用到MIUI9的研发之中。MIUI8去年5月发布，距今已有一年有余，也是时候更新换代了。当然，关于MIUI9的确切信息，我们还是等待官方消息。'
    title = '小米MIUI9首批机型曝光：共计15款'
    # contect = '新华社照片，石家庄，2017年4月17日\n一淀芦苇一淀金——雄安新区踏访记\n这是4月9日拍摄的白洋淀。\n四月的白洋淀，春风拂面，新苇吐绿。这淀芦苇，曾掩护雁翎队打鬼子，留下《小兵张嘎》等名篇；这淀芦苇，曾是养家糊口的“摇钱苇”；这淀芦苇，也曾是令人厌弃的累赘。而雄安新区的落子，又使这淀芦苇成为“无价之宝”。\n新华社记者朱旭东摄'
    # title = '（图文互动）（7）一淀芦苇一淀金——雄安新区踏访记'
    contect = '新华社广州5月9日电（记者毛一竹刘羽佳）广东省高级人民法院9日发布中英文双语《中国（广东）自由贸易试验区司法保障白皮书》。在广东自贸区挂牌成立两年来，广东高院共审结涉自贸区案件超三千件，其中涉外、涉港澳台案件占比较大。\n白皮书显示，截至2017年3月31日，南沙法院（南沙片区法院）、前海法院、横琴法院共受理和审结涉自贸区案件分别为5025件和3221件，以民商事案件居多。根据收结案及审理情况，涉自贸区案件呈现出涉外、涉港澳台案件占比较大等特点。案件当事人分布于境内外60多个国家和地区，涉诉的外商投资企业中港澳投资企业占半数以上。\n据了解，为了提高审判效率，加快涉外、涉港澳台民商事案件的办理速度，横琴法院在500多件此类案件中适用简易程序，占涉外、涉港澳台案件总数的40%以上。\n根据白皮书，各自贸区片区法院积极探索港澳籍人民陪审员参审机制，聘请港澳籍人民陪审员随机参与案件的审理和调解，增强港澳同胞对内地司法审判的参与度和认同感，提升人民法院的公信力。南沙片区法院、前海法院、横琴法院已分别选任了5名、13名、10名港澳籍人民陪审员，并邀请港澳籍陪审员参审案件，社会反响良好。(完)'

    sentences = getSentences(contect)
    title = getSentences(title)
    main(sentences, title)
    # print(list(jieba.cut(contect)))
