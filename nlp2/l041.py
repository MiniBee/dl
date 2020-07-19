import re 
import collections


train_data = '/home/peihongyue/data/code/nlp_book/4/4-16/spell-check/bayes_train_text.txt'


def words(text):
    return re.findall('[a-z]+', text.lower())

def train(features):
    model = collections.defaultdict(lambda : 1)
    for f in features:
        model[f] += 1 
    return model

word_n = train(words(open(train_data).read()))

alphabet="abcdefghijklmnopqrstuvwxyz"

def edits1(word):
    n = len(word)
    s1 = [word[0:i] + word[i+1:] for i in range(n)]
    s2 = [word[0:i] + word[i+1] + word[i] + word[i+2:] for i in range(n-1)]
    s3 = [word[0:i] + c + word[i+1:] for i in range(n) for c in alphabet]
    s4 = [word[0:i] + c + word[i:] for i in range(n) for c in alphabet]
    edits1_words = set(s1 + s2 + s3 + s4)
    edits1_words = known(edits1_words)
    return edits1_words

def edits2(word):
    edits2_words = set(e2 for e1 in edits1(word) for e2 in edits1(e1))
    return edits2_words

def known(words):
    return set([w for w in words if w in word_n])


def correct(word):
    if word not in word_n:
        candidates = known(edits1(word)) | known(edits2(word))
        return max(candidates, key=lambda w: word_n[w])
    else:
        return word 

print(correct('het'))