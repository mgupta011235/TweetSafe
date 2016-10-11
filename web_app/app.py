from flask import Flask

from flask import (request,
                   redirect,
                   url_for,
                   session,
                   render_template,
                   jsonify)

import cPickle as pickle
import gensim
import pandas as pd
import random
import numpy as np
from scipy.spatial.distance import cosine
from nltk.tokenize import PunktSentenceTokenizer

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

    simSubredditList = []

    #count how many hates there are
    for index in mostSimVecInd:
        hatecount += ishateful(docvecs.index_to_doctag(index))
        simSubredditList.append(docvecs.index_to_doctag(index))

    #majority vote to determine hateful/nothateful
    if hatecount>=threshold*len(mostSimVecInd):
        prediction = 1
    else:
        prediction = 0

    #find most similar subreddit
    mostSimSubreddit = docvecs.index_to_doctag(mostSimVecInd[0])

    return prediction,simSubredditList

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
#Flask code



# begin app

app = Flask(__name__)

# home page
@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit', methods=['POST'] )
def submission_page():
    tweet = str(request.form['newtext'])

    #load model
    modelPath = '../../doc2vec_models/basemodel2/basemodel2.doc2vec'
    model = gensim.models.Doc2Vec.load(modelPath)

    #set tuning parameters
    k = 11
    threshold = 0.6

    #find most similar subreddit
    prediction, simSubredditList = mostSimilarDoc(model,tweet,k,threshold)

    output = []

    for i in xrange(3):
        output.append(simSubredditList[i])

    page = 'The path is : {0} '
    return output[0]

    page = 'The path is : {0} '
    # sorted_similar_tweets = get_similar(tweet)
    # print sorted_similar_tweets.shape
    # print len(docs_vectorized)
    # return tweet_list[sorted_similar_tweets[0][0]]

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
