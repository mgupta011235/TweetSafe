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

##############################################################################
#main

def main():
    print "entering main..."
    modelpath = 'xgbfinal4.model'

    print "loading model..."
    model = xgb.Booster(model_file=modelpath)

    print "getting feature importances..."
    df = pd.DataFrame({'f1_score':model.get_fscore().values()},
                      index=model.get_fscore().keys())

    df.to_csv('f1_score_dataframe.csv')


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

    print "predicting..."
    proba = model.predict(xg_cv)

    dfcv['xgboost_predict'] = proba
    dfcv.to_csv('twitter_cross_val_xgboost_results.csv')

if __name__ == '__main__':
    '''This script collects feature importances and predicted probablities'''
    main()
