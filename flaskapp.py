
from flask import Flask
import gensim.downloader as api
import numpy as np
import pickle

clf = pickle.load(open('gradientBoosting.pkl', 'rb'))
wv = api.load('word2vec-google-news-300')

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Test123!'


@app.route('/Scoring<string/<string:comments>')
def Scoring(comments):
    comments_list = comments.split('///')
    scoring_comments = []
    for comment in comments_list:
        scoring_comments.append(int(predict(comment)))

    n = 0
    for i in range(len(scoring_comments)):
        if scoring_comments[i] >= 3:
            n = n + 1

    print(n)
    print(scoring_comments)
    percentage = (n / len(scoring_comments)) * 100
    return int(percentage)


def predict(comment):
    token = sent_vec(comment)
    x2d = [np.stack(token)]
    y = clf.predict(x2d)
    # print(y)
    x = str(y[0])
    return x


def sent_vec(sent):
    vector_size = wv.vector_size
    wv_res = np.zeros(vector_size)
    # print(wv_res)
    ctr = 1
    for w in sent:
        if w in wv:
            ctr += 1
            wv_res += wv[w]
    wv_res = wv_res / ctr
    return wv_res


if __name__ == '__main__':
    app.run()

