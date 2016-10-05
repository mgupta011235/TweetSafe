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

def mostSimilarDoc(model,comment,k):
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
    commentVec = model.infer_vector(wordTokens)

    #compute similarity of comment to each subreddit
    for vec_ind in xrange(len(docvecs)):
        simVals[vec_ind] = 1 - cosine(commentVec,docvecs[vec_ind])

    mostSimVecInd = np.argsort(simVals)[-k:]
    hatecount = 0

    #count how many hates there are
    for index in mostSimVecInd:
        hatecount += ishateful(docvecs.index_to_doctag(index))

    #majority vote to determine hateful/nothateful
    if hatecount>=len(mostSimVecInd):
        prediction = 1
    else:
        prediction = 0

    #find most similar subreddit
    mostSimSubreddit = docvecs.index_to_doctag(mostSimVecInd[0])

    return prediction,mostSimSubreddit

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

def train_score(model,path,numsamps,k):

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
        prediction, predictedSub = mostSimilarDoc(model,comment,k)
        predict[row] = prediction

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


def test_score(model,path,k):

    # print "loading data..."
    df = pd.read_csv(path)
    tweets = df['tweet_text'].values
    labels = df['label'].values

    predict = np.zeros((len(labels),))

    # print "scoring..."
    for row in xrange(len(labels)):
        tweet = tweets[row]
        prediction, predictedSub = mostSimilarDoc(model,tweet,k)
        predict[row] = prediction

    TP = sum(predict+labels == 2)
    TN = sum(predict+labels== 0)
    FP = sum(predict-labels == 1)
    FN = sum(predict-labels== -1)

    accu = (TP+TN)/float(len(labels))
    recall = TP/float(TP+FN)
    precision = TP/float(TP+FP)

    print ""
    print "k: {}".format(k)
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
    return [k,accu,recall,precision,TP,TN,FN,FP]


if __name__ == '__main__':

    print "starting..."

    #dataset paths
    trainpath = '../../data/labeledRedditComments.p'
    trainpath2 = '../../data/labeledRedditComments2.p'
    cvpath = '../../data/twitter_cross_val.csv'
    testpath = '../../data/twitter_test.csv'
    sqlpath = '../../data/RedditMay2015Comments.sqlite'

    #model paths
    modelPath = '../../models/basemodel2/basemodel2.doc2vec'

    print "loading model..."
    model = gensim.models.Doc2Vec.load(modelPath)

    # print "train set..."
    # train_score(model,trainpath,500,1)

    labels = ['k','accuracy','recall','precision','TP','TN','FN','FP']
    df = pd.DataFrame(columns=labels)

    print "Cross Val set..."
    for k in xrange(26):
        df.loc[k] = test_score(model,cvpath,k)
        print ""

    df.to_csv('../../data/varying_K_on_cross_val.csv')

    # print "test set..."
    # test_score(model,testpath,1)
