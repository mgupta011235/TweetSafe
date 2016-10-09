import gensim
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy.spatial.distance import cosine
from nltk.tokenize import PunktSentenceTokenizer
import time

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
#similarity code

def mostSimilarDoc(model,comment,k,threshold):
    '''
    Input: doc2vec model, comment is a str, k = number of similar doc vecs
    Output: an int indicating hate (1) or not hate (0),most similar subreddit
    '''

    docvecs = model.docvecs
    numdocvec = len(docvecs)
    simVals = np.zeros((numdocvec, ))

    #tokenize comment
    wordTokens = mytokenizer(comment)

    #create vector of tokenized comment
    #avg over 100 vectors
    finalVec = np.zeros((300, ))
    for i in xrange(100):
        finalVec = finalVec + model.infer_vector(wordTokens)
    commentVec = finalVec/100.0

    #compute similarity of comment to each subreddit
    for vec_ind in xrange(len(docvecs)):
        simVals[vec_ind] = 1 - cosine(commentVec,docvecs[vec_ind])

    mostSimVecInd = np.argsort(simVals)[-k:]
    hatecount = 0

    #count how many hates there are
    for index in mostSimVecInd:
        hatecount += ishateful(docvecs.index_to_doctag(index))

    #majority vote to determine hateful/nothateful
    if hatecount>=threshold*len(mostSimVecInd):
        prediction = 1
    else:
        prediction = 0

    #find most similar subreddit
    # mostSimSubreddit = docvecs.index_to_doctag(mostSimVecInd[0])

    return prediction

##############################################################################
#hate/NotHate code

def ishateful(subreddit):
    '''
    Input: str subreddit
    Output: int 1 if hateful subreddit, 0 otherwise
    '''

    # List of not hateful subreddits
    final_nothate_srs = ['politics', 'worldnews', 'history', 'blackladies', 'lgbt',
                         'TransSpace', 'women', 'TwoXChromosomes', 'DebateReligion',
                         'religion', 'islam', 'Judaism', 'BodyAcceptance', 'fatlogic'
                         'gaybros','AskMen','AskWomen']
    # List of hateful subreddits
    final_hateful_srs = ['CoonTown', 'WhiteRights', 'Trans_fags', 'SlutJustice',
                         'TheRedPill', 'KotakuInAction', 'IslamUnveiled', 'GasTheKikes',
                         'AntiPOZi', 'fatpeoplehate', 'TalesofFatHate','hamplanethatred',
                         'shitniggerssay','neofag','altright']

    if subreddit in final_hateful_srs:
        return 1
    else:
        return 0

#############################################################################
#scoring code

def test_score(model,path,k,threshold):

    # print "loading data..."
    df = pd.read_csv(path)
    tweets = df['tweet_text'].values
    labels = df['label'].values

    predict = np.zeros((len(labels),))

    # print "scoring..."
    for row in xrange(len(labels)):
        tweet = tweets[row]
        prediction = mostSimilarDoc(model,tweet,k,threshold)
        predict[row] = prediction

    TP = sum(predict+labels == 2)
    TN = sum(predict+labels == 0)
    FP = sum(predict-labels == 1)
    FN = sum(predict-labels == -1)

    accu = (TP+TN)/float(len(labels))
    recall = TP/float(TP+FN)
    precision = TP/float(TP+FP)

    print ""
    print "k: {}".format(k)
    print "threshold: {}".format(threshold)
    print "accuracy: {}".format(accu)
    print "recall: {}".format(recall)
    print "precision: {}".format(precision)

    print ""
    print "TP: {}".format(TP)
    print "TN: {}".format(TN)
    print ""
    print "FN: {}".format(FN)
    print "FP: {}".format(FP)

    #output data to be saved in a pd dataframe
    return [k,threshold,accu,recall,precision,TP,TN,FN,FP]

##############################################################################
#Main

if __name__ == '__main__':

    print "starting..."

    #dataset paths
    trainpath = '../../data/labeledRedditComments.p'
    trainpath2 = '../../data/labeledRedditComments2.p'
    cvpath = '../../data/twitter_cross_val.csv'
    testpath = '../../data/twitter_test.csv'
    sqlpath = '../../data/RedditMay2015Comments.sqlite'

    #model paths
    modelPath = '../../doc2vec_models/basemodel2/basemodel2.doc2vec'
    # modelPath = '../../doc2vec_models/basemodel3/basemodel3.doc2vec'
    # modelPath = '../../doc2vec_models/basemodel4/basemodel4.doc2vec'
    # modelPath = '../../doc2vec_models/modellower/modellower.doc2vec'
    # modelPath = '../../doc2vec_models/model_split/model_split.doc2vec'

    print "loading model..."
    model = gensim.models.Doc2Vec.load(modelPath)

    tstart = time.time()
    print "gridsearch..."
    results = []
    count = 0
    for k in xrange(11,12):
        for threshold in [0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69,0.7]:
            print "count: {}".format(count)
            results.append(test_score(model,cvpath,k,threshold))
            count+=1
            print ""

    labels = ['k','threshold','accuracy','recall','precision','TP','TN','FN','FP']
    df = pd.DataFrame(data=results,columns=labels)

    tstop = time.time()

    dt = tstop-tstart

    print "total time: {}".format(dt)
    print "time per gridpoint: {}".format(dt/float(count))

    df.to_csv('../../data/gridsearch_modelbase2mini_on_cross_val.csv')
