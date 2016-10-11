from sklearn.feature_extraction.text import TfidfVectorizer
import xgboost as xgb
import cPickle as pickle
from string import punctuation
from nltk import word_tokenize
from nltk.stem import snowball
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from nltk.tokenize import PunktSentenceTokenizer
import multiprocessing as mp
import time

stemmer = snowball.SnowballStemmer("english")

###############################################################################
#OHS tokenization code

def load_data(filename):
    '''
    Load data into a data frame for use in running model
    '''
    return pickle.load(open(filename, 'rb'))


def stem_tokens(tokens, stemmer):
    '''Stem the tokens.'''
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def OHStokenize(text):
    '''Tokenize & stem. Stems automatically for now.
    Leaving "stemmer" out of function call, so it works with TfidfVectorizer'''
    tokens = word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

###########################################################################
# tokenization code

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

        #if reference to twitter user, replace with usertag
        if '@' in word:
            tokenziedList.append('USER_TAG')
            continue

        #if number, replace with numtag
        #m8 is a word, 5'10" and 54-59, 56:48 are numbers
        if hasNumbers(word) and not any(char.isalpha() for char in word):
            tokenziedList.append('NUM_TAG')
            continue

        #seperate puncuations and add to tokenizedList
        newwords = seperatePunct(word).split(" ")
        tokenziedList.extend(newwords)

    return tokenziedList

def mytokenize(comment):
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

##############################################################################
#main

def search(xg_train,xg_cv,paramlist,i):
    '''Input: traing set, eval set, xgboost parameter list, index
       Output: returns a list with the parameters and the results of
               evaluating the auc of the model on the eval set
    '''

    print "start i: ", i
    print ""



    #parameters
    param = {'max_depth':paramlist[1],
             'eta':paramlist[2],
             'silent':1,
             'objective':'binary:logistic',
             'eval_metric':'auc'
             }

    num_round = paramlist[0]
    watchlist = [(xg_train, 'train'), (xg_cv, 'eval')]

    #create dictionary to store eval results
    results = dict()

    #train model
    model = xgb.train(param,
                      xg_train,
                      num_round,
                      watchlist,
                      evals_result=results,
                      verbose_eval=False)

    modelfilename = 'xgbfinal3_{}.model'.format(i)
    model.save_model(modelfilename)

    print "finish i: ", i
    print ""

    return [paramlist[0],paramlist[1],paramlist[2],results]



if __name__ == '__main__':
    '''This script runs gridsearch to determine the optimal parameters
       for xgboost on the reddit corpus'''

    print "entering main..."

    # path = '../../data/labeledRedditComments2.p'
    # cvpath = '../../data/twitter_cross_val.csv'
    #
    # load_tstart = time.time()
    # print 'loading data...'
    # df = load_data(path)
    # dfcv = pd.read_csv(cvpath)
    # load_tstop = time.time()
    #
    # #take a subset of the data for testing this code
    # randNums = np.random.randint(low=0,high=len(df.index),size=(100,1))
    # rowList = [int(row) for row in randNums]
    # dfsmall = df.ix[rowList,:]
    #
    # nf = dfsmall
    #
    # #create training set and labels
    # X = nf.body
    # y = nf.label
    #
    # Xcv = dfcv['tweet_text'].values
    # ycv = dfcv['label'].values
    #
    # vect_tstart = time.time()
    # print "vectorizing..."
    # vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
    #                        tokenizer=OHStokenize)
    #
    # # fit & transform comments matrix
    # tfidf_X = vect.fit_transform(X)
    # tfidf_Xcv = vect.transform(Xcv)
    # vect_tstop = time.time()
    #
    #
    #
    # print "converting data..."
    # #convert to dense so that DMatrix doesn't drop cols with all zeros
    # tfidf_Xcvd = tfidf_Xcv.todense()
    #
    # #data conversion to DMatrix
    # xg_train = xgb.DMatrix(tfidf_X, label=y)
    # xg_cv = xgb.DMatrix(tfidf_Xcvd, label=ycv)

    print "loading vectorizer..."
    vect = pickle.load(open('vect.p', 'rb'))

    cvpath = 'twitter_cross_val.csv'
    dfcv = pd.read_csv(cvpath)
    Xcv = dfcv['tweet_text'].values
    ycv = dfcv['label'].values

    print "transforming cross val data..."
    tfidf_Xcv = vect.transform(Xcv)
    tfidf_Xcvd = tfidf_Xcv.todense()

    xg_cv = xgb.DMatrix(tfidf_Xcvd, label=ycv)

    print "loading training data..."
    xg_train = xgb.DMatrix('xg_train2.buffer')

    print 'gridsearching...'
    grid_tstart = time.time()
    results = []
    i = 0
    for eta in [0.3,0.6,0.9]:
        for max_depth in [3,4,5]:
            for num_rounds in [100,300,600,900]:
                params = [num_rounds,max_depth,eta]
                results.append(search(xg_train,xg_cv,params,i))
                i+=1

    grid_tstop = time.time()


    #save data to dataframe
    labels = ['num_rounds','max_depth','eta','eval_results']
    df = pd.DataFrame(data=results,columns=labels)
    df.to_csv('gridsearch3_xgb.csv')

    # print "vect: {}".format(vect_tstop - vect_tstart)
    print "gridsearch: {}".format(grid_tstop - grid_tstart)
