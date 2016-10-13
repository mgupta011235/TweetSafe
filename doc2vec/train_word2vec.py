from gensim.models import Word2Vec
import pandas as pd
import cPickle as pickle
import sqlite3
import multiprocessing
import numpy as np
import time
import re
import string

#############################################################################
#Tokenizer Functions

def comment2sentence(comment):
    '''
    Convert comments to sentences & split into list of sentences
    Use re to split on multiple endings. Returns a list of separated comments.
    NOTE: currently will not strip leading whitespace.
    '''

    return re.split('[\.\?\!]', comment)


def sentence2list(sentence):
    '''
    Makes words lowercase, removes punctuation, splits words on white space & returns list.
    '''
    # make lowercase
    sentence = sentence.lower()
    # remove punctuation
    sentence = ''.join(l for l in sentence if l not in string.punctuation)

    # return split sentence in a list
    return sentence.split()


def prepare_comments(comments):
    '''
    Prepare a set of comments for reading in & processing by word2vec
    '''

    fullcommentslist = []

    # go through each comment
    for comment in comments:
        # break comments into list of sentences
        listofsent = comment2sentence(comment)
        # split each sentence into a list of words
        for sent in listofsent:
            listofwordsofsent = sentence2list(sent)
            fullcommentslist.append(listofwordsofsent)

    return fullcommentslist

#############################################################################
#Generator Functions

def df_gen(df):
    '''
    Input: a pandas df
    Output: this is a generator that gives the next row in the df
    '''

    numrows = len(df.index)
    for row in xrange(numrows):

        comment = df.iloc[row,:]
        body = comment['body']
        subreddit = str(comment['subreddit'])

        # Clean and tokenize text
        body = prepare_comments(body)

        # generate
        # print "{}: {}".format(numrows,row)
        # print "{}: {}".format(subreddit,body)
        # print ""
        # yield LabeledSentence(body,tags=[subreddit])

        #yield a list for word2vec
        yield body

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
            body = prepare_comments(body)
            # yield LabeledSentence(body,labels=['subreddit'])
            yield body
        except:
            yield []

##############################################################################
#Model Functions

def build_model(gen_obj):
    '''Input: a generator source to generate training data from
       Output: a trained doc2vec models
       '''

    workers = multiprocessing.cpu_count()
    print "workers: {}".format(workers)


    print "creating sentence list from generator"
    sentence = []

    generate_tstart = time.time()
    for comment in df_gen(gen_obj):
        sentence.append(comment)
    generate_tstop = time.time()

    print "generate: {}".format(generate_tstop - generate_tstart)

    print "training model..."
    w2v_reddit_model = Word2Vec( sentences=sentence,
                                size=300,
                                window=15,
                                negative=5,
                                hs=0,
                                min_count=2,
                                sample=1e-5,
                                workers=workers)


    return w2v_reddit_model

###############################################################################
#Main

if __name__ == '__main__':
    ''' This script trains two word2vec models, one on hateful subreddits
        the other on not hateful subreddits'''


    print "starting..."

    path = 'hateComments.p'
    path1 = 'nothateComments.p'



    print "loading dataframes..."
    t_load_df_start = time.time()
    df = pickle.load(open(path, 'rb'))
    df1 = pickle.load(open(path1, 'rb'))
    t_load_df_stop = time.time()


    # print "connecting to sql database..."
    # conn = sqlite3.connect(path2)
    # c = conn.cursor()
    # c.execute("SELECT subreddit, body FROM MAY2015")
    # mygen = sql_gen(c)

    print "building hate model..."
    t_build_model_start = time.time()
    model = build_model(df)
    t_build_model_stop = time.time()

    print "saving hate model..."
    model.save('hate_model2.word2vec')


    print "build_model: {}".format(t_build_model_stop - t_build_model_start)

    print "building nothate model..."
    t_build_model_start = time.time()
    model = build_model(df1)
    t_build_model_stop = time.time()

    print "saving nothate model..."
    model.save('nothate_model2.word2vec')

    print "build nothate model: {}".format(t_build_model_stop - t_build_model_start)

    print "load df: {}".format(t_load_df_stop - t_load_df_start)
