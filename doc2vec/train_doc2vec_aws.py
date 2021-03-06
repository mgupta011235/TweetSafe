from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec
from nltk.tokenize import PunktSentenceTokenizer
import pandas as pd
import cPickle as pickle
import sqlite3
import multiprocessing
import numpy as np
import time

#############################################################################
#Tokenizer Functions

def seperatePunct(incomingString):
    '''
    Input:str,
    Output: str with all puncuations seperated by spaces
    '''
    outstr = ''
    characters = set(['!','@','#','$',"%","^","&","*",":","\\",
                  "(",")","+","=","?","\'","\"",";","/",
                  "{","}","[","]","<",">","~","`","|"])

    for char in incomingString:
        if char in characters:
            outstr = outstr + ' ' + char + ' '
        else:
            outstr = outstr + char

    return outstr

def hasNumbers(inputString):
    '''
    Input: str
    Output: returns a 1 if the string contains a number
    '''
    return any(char.isdigit() for char in inputString)

def text_cleaner(wordList):
    '''
    INPUT: List of words to be tokenized
    OUTPUT: List of tokenized words
    '''

    tokenziedList = []

    for word in wordList:

        #remove these substrings from the word
        word = word.replace('[deleted]','')
        word = word.replace('&gt','')

        #if link, replace with linktag
        if 'http' in word:
            tokenziedList.append('LINK_TAG')
            continue

        #if reference to subreddit, replace with reddittag
        if '/r/' in word:
            tokenziedList.append('SUBREDDIT_TAG')
            continue

        #if reference to reddit user, replace with usertag
        if '/u/' in word:
            tokenziedList.append('USER_TAG')
            continue

        # #if reference to twitter user, replace with usertag
        # if '@' in word:
        #     tokenziedList.append('USER_TAG')
        #     continue

        #if number, replace with numtag
        #m8 is a word, 5'10" and 54-59, 56:48 are numbers
        if hasNumbers(word) and not any(char.isalpha() for char in word):
            tokenziedList.append('NUM_TAG')
            continue

        #seperate puncuations and add to tokenizedList
        newwords = seperatePunct(word).split(" ")
        tokenziedList.extend(newwords)

    return tokenziedList

def mytokenizer(comment):
    '''
    Input: takes in a reddit comment as a str or unicode and tokenizes it
    Output: a tokenized list
    '''

    tokenizer = PunktSentenceTokenizer()
    sentenceList = tokenizer.tokenize(comment)
    wordList = []
    for sentence in sentenceList:
        wordList.extend(sentence.split(" "))

    return text_cleaner(wordList)

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
        body = mytokenizer(body)

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
            body = mytokenizer(body)
            yield LabeledSentence(body,labels=['subreddit'])
        except:
            yield []

##############################################################################
#Model Functions

def build_model(gen_obj):
    '''Input: a generator source to generate training data from
       Output: a trained doc2vec model'''

    cores = multiprocessing.cpu_count()
    # print "FAST_VERSION: {}".format(gensim.models.doc2vec.FAST_VERSION)
    # assert gensim.models.doc2vec.FAST_VERSION > -1
    # print "FAST_VERSION: {}".format(gensim.models.doc2vec.FAST_VERSION)
    print "cores: {}".format(cores)

    workers = cores
    print "workers: {}".format(workers)


    d2v_reddit_model = Doc2Vec( dm=0,
                                size=300,
                                window=15,
                                negative=5,
                                hs=0,
                                min_count=2,
                                sample=1e-5,
                                workers=workers)

# model below was used for testing script
    # d2v_reddit_model = Doc2Vec( dm=0,
    #                             size=3,
    #                             window=3,
    #                             workers=cores)

    print "building vocabulary..."
    t_build_vocab_start = time.time()
    d2v_reddit_model.build_vocab(df_gen(gen_obj)) #sentence_gen(reddit_data))
    t_build_vocab_stop = time.time()


    print "training model..."
    t_train_model_start = time.time()
    for epoch in xrange(20):
        print "epoch: {}".format(epoch)
        d2v_reddit_model.train(df_gen(gen_obj))
        d2v_reddit_model.alpha -= 0.002  # decrease the learning rate
        d2v_reddit_model.min_alpha = d2v_reddit_model.alpha  # fix the learning rate, no decay
    t_train_model_stop = time.time()

    print "build_vocab: {}".format(t_build_vocab_stop - t_build_vocab_start)
    print "train_model: {}".format(t_train_model_stop - t_train_model_start)

    return d2v_reddit_model

###############################################################################
#Main

if __name__ == '__main__':
    '''This script trains a doc2vec model'''
    print "starting..."

    path = 'labeledRedditComments2.p'
    path1 = 'labeledRedditComments.p'
    path2 = 'data/RedditMay2015Comments.sqlite'


    print "loading dataframe..."
    t_load_df_start = time.time()
    df = pickle.load(open(path, 'rb'))
    t_load_df_stop = time.time()


    # randNums = np.random.randint(low=0,high=len(df.index),size=(200000,1))
    # rowList = [int(row) for row in randNums]
    # dfsmall = df.ix[rowList,:]


    # print "connecting to sql database..."
    # conn = sqlite3.connect(path2)
    # c = conn.cursor()
    # c.execute("SELECT subreddit, body FROM MAY2015")
    # mygen = sql_gen(c)

    print "building model..."
    t_build_model_start = time.time()
    model = build_model(df)
    t_build_model_stop = time.time()

    print "load df: {}".format(t_load_df_stop - t_load_df_start)
    print "build_model: {}".format(t_build_model_stop - t_build_model_start)

    print "saving model..."
    model.save('models/basemodel3.doc2vec')
