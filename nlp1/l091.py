import numpy as np
import jieba
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression


def get_data():
    with open('/Users/peihongyue/phy/project/dl/data/email/ham_data.txt') as ham_f, open('/Users/peihongyue/phy/project/dl/data/email/spam_data.txt') as spam_f:
        ham_data = ham_f.readlines()
        spam_data = spam_f.readlines()
        ham_label = np.ones(len(ham_data)).tolist()
        spam_label = np.zeros(len(spam_data)).tolist()
        corpus = ham_data + spam_data
        labels = ham_label + spam_label
    return corpus, labels


def prepare_data(corpus, labels, test_data_proportion=0.3):
    x_train, x_test, y_train, y_test = train_test_split(corpus, labels, test_size=test_data_proportion)
    return x_train, x_test, y_train, y_test


def tokenize_text(text):
    tokens = jieba.cut(text)
    tokens = [token.strip() for token in tokens]
    return tokens


def remove_special_characters(text):
    tokens = tokenize_text(text)
    pattern = re.compile('[{}]'.format(re.escape(string.punctuation)))
    filterd_tokens = filter(None, [pattern.sub('', token) for token in tokens])
    filterd_tokens = remove_stopwords(filterd_tokens)
    filterd_text = ''.join(filterd_tokens)
    return filterd_text


def remove_stopwords(tokens):
    with open('/Users/peihongyue/phy/project/dl/data/email/stop_words.utf8') as f:
        stopword_list = f.readlines()
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    return filtered_tokens


def normalize_corpus(corpus, tokenize=False):
    normalized_corpus = []
    for text in corpus:
        text = remove_special_characters(text)
        if tokenize:
            text = tokenize_text(text)
            normalized_corpus.append(text)
        else:
            normalized_corpus.append(text)
    return normalized_corpus


def bow_extractor(corpus, ngram_range=(1, 1)):
    vectorizer = CountVectorizer(min_df=1, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def tfidf_transformer(bow_matrix):
    transformer = TfidfTransformer(norm='l2', smooth_idf=TfidfTransformer, use_idf=TfidfTransformer)
    tfidf_matrix = transformer.fit_transform(bow_matrix)
    return transformer, tfidf_matrix


def tfidf_extractor(corpus, ngram_range=(1,1)):
    vectorizer = TfidfVectorizer(min_df=1, norm='l2', smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features


def remove_empty_docs(corpus, labels):
    filtered_corpus = []
    filtered_labels = []
    for doc, label in zip(corpus, labels):
        if doc.strip():
            filtered_corpus.append(corpus)
            filtered_labels.append(labels)
    return filtered_corpus, filtered_labels


def get_metrics(true_labels, predicted_labels):
    print('准确率:', np.round(metrics.accuracy_score(true_labels, predicted_labels), 2))
    print('精度:', np.round(metrics.precision_score(true_labels, predicted_labels), 2))
    print('召回率:', np.round(metrics.recall_score(true_labels, predicted_labels), 2))
    print('F1得分:', np.round(metrics.f1_score(true_labels, predicted_labels), 2))


def train_predict_evaluate_model(classifier, train_features, train_labels, test_features, test_labels):
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)
    get_metrics(test_labels, predictions)
    return predictions


def main():
    corpus, labels = get_data()
    # print(corpus[:10])
    # print(labels[:10])
    print('数据总量:', len(labels), len(corpus))
    train_corpus, test_corpus, train_labels, test_labels = prepare_data(corpus, labels, test_data_proportion=0.3)
    norm_train_corpus = normalize_corpus(train_corpus)
    norm_test_corpus = normalize_corpus(test_corpus)

    bow_vectorizer, bow_train_features = bow_extractor(norm_train_corpus)
    bow_test_features = bow_vectorizer.transform(norm_test_corpus)

    tfidf_vectorizer, tfidf_train_features = tfidf_extractor(norm_train_corpus)
    tfidf_test_features = tfidf_vectorizer.transform(norm_test_corpus)

    tokenized_train = [jieba.lcut(text) for text in norm_train_corpus]
    tokenized_test = [jieba.lcut(text) for text in norm_test_corpus]

    mnb = MultinomialNB()
    svm = SGDClassifier(loss='hinge', max_iter=100)
    lr = LogisticRegression()

    mnb_bow_predictions = train_predict_evaluate_model(classifier=mnb, train_features=bow_train_features,
                                                       train_labels=train_labels, test_features=bow_test_features,
                                                       test_labels=test_labels)

    lr_bow_predictions = train_predict_evaluate_model(classifier=lr, train_features=bow_train_features,
                                                       train_labels=train_labels, test_features=bow_test_features,
                                                       test_labels=test_labels)

    svm_bow_predictions = train_predict_evaluate_model(classifier=svm, train_features=bow_train_features,
                                                      train_labels=train_labels, test_features=bow_test_features,
                                                      test_labels=test_labels)

    label_name_map = ["垃圾邮件", "正常邮件"]
    num = 0
    for document, label, predicted_label in zip(test_corpus, test_labels, svm_bow_predictions):
        if label == 0 and predicted_label == 0:
            print('邮件类型:', label_name_map[int(label)])
            print('预测的邮件类型:', label_name_map[int(predicted_label)])
            print('文本:-')
            print(re.sub('\n', ' ', document))

            num += 1
            if num == 4:
                break


if __name__ == '__main__':
    main()


