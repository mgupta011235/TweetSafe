from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec
from nltk import download
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import pandas as pd
import cPickle as pickle
import sqlite3
import multiprocessing
import numpy as np


def text_cleaner(text):
    '''
    INPUT: string of body text
    OUTPUT: List of tokenized lower case words with stopwords removed
    '''

    # Output tokenizes text and removes any stopwords and then outptus lowercased words
    return [word.lower() for word in tokenizer.tokenize(text) if not word.lower() in stopwords]


def df_gen(df):
    '''
    Input: a pandas df
    Output: this is a generator that gives the next row in the df
    '''

    numrows = len(df.index)
    for row in xrange(numrows):

        # try:
        #     # load a comment
        #     comment = df.iloc[row,:]
        #     body = comment['body']
        #     subreddit = str(comment['subreddit'])
        #
        #     # Clean and tokenize text
        #     body = text_cleaner(body)
        #
        #     # generate
        #     yield LabeledSentence(body,labels=[str(json_object['subreddit'])])
        # except:
        #     yield None

        comment = df.iloc[row,:]
        body = comment['body']
        subreddit = str(comment['subreddit'])

        # Clean and tokenize text
        body = text_cleaner(body)

        # generate
        # print "{}: {}".format(numrows,row)
        # print "{}: {}".format(subreddit,body)
        # print ""
        yield LabeledSentence(body,tags=[subreddit])

def sql_gen(c):
    '''
    Input: sqlite3 cursor to a sqlite3 database
    Output: this is a generator that gives the next query result from c
    '''

    # c is generated using the following code
    # conn = sqlite3.connect(path2)
    # c = conn.cursor()
    # c.execute("SELECT subreddit, body FROM MAY2015")

    for comment in c:
        try:
            subreddit = str(comment[0])
            body = comment[1]
            yield LabeledSentence(body,labels=[str(json_object['subreddit'])])
        except:
            yield None


def build_model(gen_obj):

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1


    # d2v_reddit_model = Doc2Vec( dm=0,
    #                             size=300,
    #                             window=15,
    #                             negative=5,
    #                             hs=0,
    #                             min_count=2,
    #                             sample=1e-5,
    #                             workers=cores)

# model below was used for testing script
    d2v_reddit_model = Doc2Vec( dm=0,
                                size=3,
                                window=3,
                                workers=cores)
    print "building vocabulary..."
    d2v_reddit_model.build_vocab(gen_obj) #sentence_gen(reddit_data))

    print "training model..."
    for epoch in xrange(1):
        print "epoch: {}".format(epoch)
        d2v_reddit_model.train(gen_obj)
        d2v_reddit_model.alpha -= 0.002  # decrease the learning rate
        d2v_reddit_model.min_alpha = d2v_reddit_model.alpha  # fix the learning rate, no decay

    return d2v_reddit_model


if __name__ == '__main__':
    print "downloading all corpora from nltk..."
    # download('all-corpora')

    stopwords = stopwords.words('english')
    stopwords.extend(['[deleted]','[removed]'])
    tokenizer = RegexpTokenizer(r'\w+')

    path1 = '../../data/labeledRedditComments.p'
    path2 = '../../data/RedditMay2015Comments.sqlite'

    print "creating generator..."

    print "loading dataframe..."
    df = pickle.load(open(path1, 'rb'))
    # dfsmall = df.ix[:100,:]
    mygen = df_gen(df)

    # print "connecting to sql database..."
    # conn = sqlite3.connect(path2)
    # c = conn.cursor()
    # c.execute("SELECT subreddit, body FROM MAY2015")
    # mygen = sql_gen(c)

    print "building model..."
    model = build_model(mygen)

    print "saving model..."
    model.save('my_model.doc2vec')
