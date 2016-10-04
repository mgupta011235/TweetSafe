from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec
from nltk import download
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import stopwords
import pandas as pd
import cPickle as pickle
import sqlite3
import multiprocessing
import numpy as np
import time

def cleanString(incomingString):
    '''
    INPUT: string
    OUTPUT: string with bad chars removed
    '''

    newstring = incomingString

    #remove these chars
    newstring = newstring.replace(",","")
    newstring = newstring.replace("-","")
    newstring = newstring.replace("!","")
    newstring = newstring.replace("@","")
    newstring = newstring.replace("#","")
    newstring = newstring.replace("$","")
    newstring = newstring.replace("%","")
    newstring = newstring.replace("^","")
    newstring = newstring.replace("&","")
    newstring = newstring.replace("*","")
    newstring = newstring.replace("(","")
    newstring = newstring.replace(")","")
    newstring = newstring.replace("+","")
    newstring = newstring.replace("=","")
    newstring = newstring.replace("?","")
    newstring = newstring.replace("\'","")
    newstring = newstring.replace("\"","")
    newstring = newstring.replace("{","")
    newstring = newstring.replace("}","")
    newstring = newstring.replace("[","")
    newstring = newstring.replace("]","")
    newstring = newstring.replace("<","")
    newstring = newstring.replace(">","")
    newstring = newstring.replace("~","")
    newstring = newstring.replace("`","")
    newstring = newstring.replace(":","")
    newstring = newstring.replace(";","")
    newstring = newstring.replace("|","")
    newstring = newstring.replace("\\","")
    newstring = newstring.replace("/","")

    return newstring

def tokenizePunc(incomingString):
    '''
    INPUT: string
    OUTPUT: string with spaces added between puncuations
    '''

    newstring = incomingString

    #tokenize these
    newstring = newstring.replace("."," . ")
    newstring = newstring.replace(","," , ")
    newstring = newstring.replace("!"," !")
    newstring = newstring.replace("(","( ")
    newstring = newstring.replace(")"," )")
    newstring = newstring.replace("?"," ? ")
    newstring = newstring.replace("\""," \" ")
    newstring = newstring.replace(":"," : ")
    newstring = newstring.replace(";"," ; ")
    newstring = newstring.replace("*"," * ")
    newstring = newstring.replace("+"," + ")
    newstring = newstring.replace("="," = ")
    newstring = newstring.replace("{"," { ")
    newstring = newstring.replace("}"," } ")
    newstring = newstring.replace("["," [ ")
    newstring = newstring.replace("]"," ] ")
    newstring = newstring.replace("<"," < ")
    newstring = newstring.replace(">"," > ")
    newstring = newstring.replace("~"," ~ ")
    newstring = newstring.replace("|"," | ")
    newstring = newstring.replace("/"," / ")
    newstring = newstring.replace("\\"," \\ ")

    return newstring


def text_cleaner(wordList):
    '''
    INPUT: list of words
    OUTPUT: List of tokenized lower case words with stopwords removed
    '''

    badSubStringList = ['[deleted]','&','/r/','/u/']
    cleanedList = []

    for word in wordList:

        #if the word has a bad substring, dont add it to the output
        if any(substring in word for substring in badSubStringList):
            continue

        #if the word is a number, replace it with a num tag
        try:
            newstring = cleanString(word) #5'10, --> 510
            val = float(newstring)
            cleanedList.append('NUMTAG')
            continue
        except:
            pass

        #if a word is a link, replace it with a link tag
        if 'http://' in word:
            cleanedList.append('LINKTAG')
            continue


        #tokenize puncuations and remove unwanted chars
        newwords = tokenizePunc(word).split()

        cleanedList.extend(newwords)

    return cleanedList

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

        # tokenize text
        sentenceList = tokenizer.tokenize(body)
        wordList = []
        for sentence in sentenceList:
            wordList.extend(sentence.split())

        #clean text
        body = text_cleaner(wordList)

        # generate
        # print "{}: {}".format(numrows,row)
        # print "{}: {}".format(subreddit,body)
        # print body
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
        # try:
        #     subreddit = str(comment[0])
        #     body = comment[1]
        #     yield LabeledSentence(body,tags=[subreddit])
        # except:
        #     yield None

        subreddit = str(comment[0])

        # tokenize text
        sentenceList = tokenizer.tokenize(comment[1])
        wordList = []
        for sentence in sentenceList:
            wordList.extend(sentence.split())

        #clean text
        body = text_cleaner(wordList)

        # generate
        yield LabeledSentence(body,tags=[subreddit])

def build_model(gen_obj):

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1
    print "cores: ".format(cores)

    d2v_reddit_model = Doc2Vec( dm=0,
                                size=300,
                                window=15,
                                negative=5,
                                hs=0,
                                min_count=2,
                                sample=1e-5,
                                workers=1)

# model below was used for testing script
    # d2v_reddit_model = Doc2Vec( dm=0,
    #                             size=3,
    #                             window=3,
    #                             workers=cores)

    t_build_vocab_start = time.time()
    print "building vocabulary..."
    d2v_reddit_model.build_vocab(gen_obj)
    t_build_vocab_stop = time.time()

    t_train_model_start = time.time()
    print "training model..."
    for epoch in xrange(20):
        print "epoch: {}".format(epoch)
        d2v_reddit_model.train(gen_obj)
        d2v_reddit_model.alpha -= 0.002  # decrease the learning rate
        d2v_reddit_model.min_alpha = d2v_reddit_model.alpha  # fix the learning rate, no decay
    t_train_model_stop = time.time()

    print "build vocab: {}".format(t_build_vocab_stop - t_build_vocab_start)
    print "train model: {}".format(t_train_model_stop - t_train_model_start)

    return d2v_reddit_model


if __name__ == '__main__':
    print "starting..."
    # print "downloading all corpora from nltk..."
    # download('all-corpora')

    # stopwords = stopwords.words('english')
    tokenizer = PunktSentenceTokenizer()

    path1 = 'labeledRedditComments.p'
    path2 = '../../data/RedditMay2015Comments.sqlite'



    print "loading dataframe..."
    t_load_df_start = time.time()
    df = pickle.load(open(path1, 'rb'))
    t_load_df_stop = time.time()


    #select random rows to create a random df matrix
    randRows = np.random.randint(low=0,high=len(df.index),size=(200000,1))
    rows = [int(row) for row in randRows]
    dfsmall = df.ix[rows,:]

    print "creating generator..."
    mygen = df_gen(dfsmall)

    # print "connecting to sql database..."
    # conn = sqlite3.connect(path2)
    # c = conn.cursor()
    # c.execute("SELECT subreddit, body FROM MAY2015")
    # mygen = sql_gen(c)
    t_build_model_start = time.time()
    print "building model..."
    model = build_model(mygen)
    t_build_model_stop = time.time()

    print "load df: {}".format(t_load_df_stop - t_load_df_start)
    print "build model: {}".format(t_build_model_stop - t_build_model_start)

    print "saving model..."
    model.save('my_model.doc2vec')
