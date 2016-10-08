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

def main():
    print "entering main..."

    path = '../../data/labeledRedditComments2.p'
    cvpath = '../../data/twitter_cross_val.csv'

    load_tstart = time.time()
    print 'loading data...'
    df = load_data(path)
    dfcv = pd.read_csv(cvpath)
    load_tstop = time.time()

    #take a subset of the data for testing this code
    randNums = np.random.randint(low=0,high=len(df.index),size=(200,1))
    rowList = [int(row) for row in randNums]
    dfsmall = df.ix[rowList,:]

    nf = dfsmall

    #create training set and labels
    X = nf.body
    y = nf.label

    Xcv = dfcv['tweet_text'].values
    ycv = dfcv['label'].values

    vect_tstart = time.time()
    print "vectorizing..."
    vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
                           tokenizer=mytokenize)


    # fit & transform comments matrix
    tfidf_X = vect.fit_transform(X)
    tfidf_Xcv = vect.transform(Xcv)
    vect_tstop = time.time()

    print "tfidf_X.shape {}".format(tfidf_X.shape)
    print "tfidf_Xcv.shape {}".format(tfidf_Xcv.shape)


    # develop data to train model
    # .todense() b/c otherwise DMatrix conversion will drop cols b/c in a
    # sparse matrix, a col with all 0's doesn't exist
    xg_train = xgb.DMatrix(tfidf_X, label=y)
    xg_cv = xgb.DMatrix(tfidf_Xcv.todense(), label=ycv)

    train_tstart = time.time()
    print 'training...'

    #parameters
    param = {'max_depth':2,
             'eta':0.5,
             'silent':1,
             'objective':'binary:logistic',
             'eval_metric':'auc'
             }
    #number of boosted rounds
    num_round = 1

    # what to apply the eval metric to
    # what the eval metric on these as you train to obj
    watchlist = [(xg_train, 'train'), (xg_cv, 'eval')]

    #dict with the results of the model on the eval_metric
    results = dict()

    #train model
    model = xgb.train(param,
                      xg_train,
                      num_round,
                      watchlist,
                      evals_result=results, #store eval results in results dic
                      verbose_eval=False)   #dont print output to screen
    train_tstop = time.time()

    print "saving model..."
    model.save_model('../../xgb_models/xgb.model')

    print "load data: {}".format(load_tstop - load_tstart)
    print "tfidf: {}".format(vect_tstop - vect_tstart)
    print "train: {}".format(train_tstop - train_tstart)


    # To load saved model:
    # model = xgb.Booster(model_file='../../xgb_models/xgb.model')

if __name__ == '__main__':
    main()
