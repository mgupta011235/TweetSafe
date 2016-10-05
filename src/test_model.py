import gensim
import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
import cPickle as pickle
from scipy.spatial.distance import cosine
from nltk.tokenize import PunktSentenceTokenizer

###########################################################################
# tokenization code

def seperatePunct(incomingString):
    newstring = incomingString
    newstring = newstring.replace("!"," ! ")
    newstring = newstring.replace("@"," @ ")
    newstring = newstring.replace("#"," # ")
    newstring = newstring.replace("$"," $ ")
    newstring = newstring.replace("%"," % ")
    newstring = newstring.replace("^"," ^ ")
    newstring = newstring.replace("&"," & ")
    newstring = newstring.replace("*"," * ")
    newstring = newstring.replace("("," ( ")
    newstring = newstring.replace(")"," ) ")
    newstring = newstring.replace("+"," + ")
    newstring = newstring.replace("="," = ")
    newstring = newstring.replace("?"," ? ")
    newstring = newstring.replace("\'"," \' ")
    newstring = newstring.replace("\""," \" ")
    newstring = newstring.replace("{"," { ")
    newstring = newstring.replace("}"," } ")
    newstring = newstring.replace("["," [ ")
    newstring = newstring.replace("]"," ] ")
    newstring = newstring.replace("<"," < ")
    newstring = newstring.replace(">"," > ")
    newstring = newstring.replace("~"," ~ ")
    newstring = newstring.replace("`"," ` ")
    newstring = newstring.replace(":"," : ")
    newstring = newstring.replace(";"," ; ")
    newstring = newstring.replace("|"," | ")
    newstring = newstring.replace("\\"," \\ ")
    newstring = newstring.replace("/"," / ")
    return newstring

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
        if 'http://' in word:
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

def mostSimilarDoc(model,comment):
    '''
    Input: doc2vec model, comment is a str
    Output: the label of the doc most similar to the comment
    '''

    docvecs = model.docvecs

    wordTokens = mytokenizer(comment)
    commentVec = model.infer_vector(wordTokens)

    mostSimVec = None
    bestSimVal = None

    for vec_ind in xrange(len(docvecs)):
        simVal = 1 - cosine(commentVec,docvecs[vec_ind])

        if simVal>bestSimVal:
            mostSimVec = vec_ind
            bestSimVal = simVal

    return docvecs.index_to_doctag(mostSimVec), bestSimVal

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

##############################################################################
#testing code

def train_score(model,path,numsamps):

    print "loading data..."
    df = pickle.load(open(path, 'rb'))

    if not numsamps:
        numsamps = len(df.index)

    randrows = random.sample(xrange(len(df.index)), numsamps)
    comments = df.ix[randrows,'body'].values
    subreddits = df.ix[randrows,'subreddit'].values
    subredditScore = 0

    labels = df['label'].values
    predict = np.zeros((len(labels),))

    print "scoring..."
    for row,comment in enumerate(comments):
        predictedSub, simVal = mostSimilarDoc(model,comment)
        predict[row] = ishateful(predictedSub)

        if predictedSub == subreddits[row]:
            subredditScore += 1

    print ""
    print "subredditScore: {}".format(subredditScore/float(numsamps))

    TP = sum(predict+labels == 2)
    TN = sum(predict+labels== 0)
    FP = sum(predict-labels == 1)
    FN = sum(predict-labels== -1)

    accu = (TP+TN)/float(len(labels))
    recall = TP/float(TP+FN)
    precision = TP/float(TP+FP)

    print ""
    print "accuracy: {}".format(accu)
    print "recall: {}".format(recall)
    print "precision: {}".format(precision)

    print ""
    print "TP: {}".format(TP)
    print "TN: {}".format(TN)
    print ""
    print "FN: {}".format(FN)
    print "FP: {}".format(FP)


def test_score(model,path):

    print "loading data..."
    df = pd.read_csv(path)
    tweets = df['tweet_text'].values
    labels = df['label'].values

    predict = np.zeros((len(labels),))

    print "scoring..."
    for row in xrange(len(labels)):
        tweet = tweets[row]
        predictedSub, simVal = mostSimilarDoc(model,tweet)
        predict[row] = ishateful(predictedSub)

    TP = sum(predict+labels == 2)
    TN = sum(predict+labels== 0)
    FP = sum(predict-labels == 1)
    FN = sum(predict-labels== -1)

    accu = (TP+TN)/float(len(labels))
    recall = TP/float(TP+FN)
    precision = TP/float(TP+FP)

    print ""
    print "accuracy: {}".format(accu)
    print "recall: {}".format(recall)
    print "precision: {}".format(precision)

    print ""
    print "TP: {}".format(TP)
    print "TN: {}".format(TN)
    print ""
    print "FN: {}".format(FN)
    print "FP: {}".format(FP)


if __name__ == '__main__':

    print "starting..."

    trainpath = '../../data/labeledRedditComments.p'
    trainpath2 = '../../data/labeledRedditComments2.p'
    testpath = '../../data/twitter-hate-speech-classifier-clean.csv'
    path3 = '../../data/RedditMay2015Comments.sqlite'
    modelPath = '../../models/base_model_original_tokenizer/base_model_original_tokenizer.doc2vec'

    print "loading model..."
    model = gensim.models.Doc2Vec.load(modelPath)

    print "train set..."
    train_score(model,trainpath,None)

    # print "test set..."
    # test_score(model,testpath)
