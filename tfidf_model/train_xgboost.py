#Online hate speech libraries
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

#My libraries
from nltk.tokenize import PunktSentenceTokenizer
import sqlite3
import time


##############################################################################
#Online hate speech tokenization process

stemmer = snowball.SnowballStemmer("english")

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

##############################################################################
#my tokenization process

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


def main():

    path = '../../data/labeledRedditComments2.p'

    print 'Loading Data'
    df = load_data(path)
    X = df.body
    y = df.label


    print "Vectorizing"
    vect = TfidfVectorizer(stop_words='english', decode_error='ignore',
                           tokenizer=OHStokenize)

    # fit & transform comments matrix
    tfidf_X = vect.fit_transform(X)

    # Save out vect & tfidf_X
    pickle.dump(vect, open('vect.p', 'wb'))
    pickle.dump(tfidf_X, open('tfidf_X.p', 'wb'))

    # develop data to train model
    xg_train = xgb.DMatrix(tfidf_X, label=y)

    print 'Classifying'
    # Set up xboost parameters
    # use softmax multi-class classification to return probabilities
    param = {'objective': 'multi:softprob',
             'eta': 0.9,
             'max_depth': 6,
             }

    watchlist = [(xg_train, 'train')]
    num_round = 1  # Number of rounds determined after running cross validation
    model = xgb.train(param, xg_train, num_round, watchlist)

    # pickling just in case
    pickle.dump(model, open('../../xgb_models/xgb_model.p', 'wb'))

    #save model
    model.save_model('xgb.model')

    #dump model
    model.dump_model('dump.raw.txt', 'featmap.txt')


    # To load saved model:
    # model = xgb.Booster({'nthread': 4}) #init model
    # model.load_model("model.bin")  <-- I think this should be "hatespeech.model"

if __name__ == '__main__':
    main()
