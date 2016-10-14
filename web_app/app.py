from flask import Flask

from flask import (request,
                   redirect,
                   url_for,
                   session,
                   render_template,
                   jsonify)

import gensim
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




#############################################################################
#
def filterinput(tweet):
    '''checks the input for bad chars and replaces them
       with a space instead. '''
    out = ""
    for character in tweet:
        try:
            val = str(character)
            out = out + val
        except:
            out = out + " "
    return out



##############################################################################
#Flask code



# begin app

app = Flask(__name__)

# home page
@app.route('/')
def welcome():
    '''Home page'''
    return render_template('index.html')

@app.route('/about')
def about():
    '''this page describes my project'''

    paragraph0 = "Why TweetSafe?"

    paragraph1 = ''' Online communities such as twitter have grown to
    become important centers where everyone can exchange information and ideas.
    All too often however, people use these sites as a platform to attack and
    dehumanize others. Because of the huge information flow on these sites, it's
    impossible for human administrators to effectively police these abusive users.
    To solve this problem I developed TweetSafe. TweetSafe is a neural network
    that determines if a tweet is offensive or not. TweetSafe has the ability to
    sift through millions of tweets in mintues and flag abusive users. With this
    tool Twitter will be able to effciently track and ban abusive users from its website. '''

    paragraph2 = "How does it work?"

    paragraph3 = '''TweetSafe is built on a neural network architecture called
    Doc2Vec. Doc2Vec learns relationships between words and then maps each word to
    a unique vector. These vector representations of words (called word embeddings)
    are than used to find vector representations of document labels. In this case,
    Doc2Vec was trained on 1.8 million reddit comments. The document labels
    are subreddits and the words are comments from those subreddits. What TweetSafe
    does is it takes a tweet, converts it into a vector and finds the 11 most similar
    subreddits using cosine similarity. If 63% of those subreddits are offensive
    subreddits, than the tweet is labeled as offensive'''

    paragraph4 = "Beyond TweetSafe"

    paragraph5 = '''While TweetSafe is tuned to detect abusive language
    on Twitter, the method I outlined above can be applied to any online community.
    I'm making this project avilible to anyone who is interested. I hope that by
    doing so others will be inpsired by my work and continue to improve upon it.
    If you do decide to use my work I encourage you to send me your results. I
    would love to see what ideas you came up with!  '''

    paragraphs = [paragraph0,"", paragraph1, "", paragraph2,"", paragraph3,"", paragraph4, "", paragraph5]

    return render_template('index3.html',paragraphs=paragraphs)

@app.route('/submit', methods=['POST'] )
def submission_page():
    '''outputs the 11 most similar subreddits and color codes the text.
       red for offensive, green for not offensive'''
    # try:
    #     tweet = str(request.form['newtext'])
    # except UnicodeEncodeError:
    #     tweet = "Oops! That tweet contained a UnicodeEncodeError. Please try another tweet."

    #filter the input for bad chars
    tweet = filterinput(request.form['newtext'])

    #load model
    modelPath = '../../doc2vec_models/basemodel2/basemodel2.doc2vec'
    model = gensim.models.Doc2Vec.load(modelPath)

    #set tuning parameters
    k = 11
    threshold = 0.63

    #find most similar subreddit
    prediction, simSubredditList = mostSimilarDoc(model,tweet,k,threshold)

    #a list of the classes of the subreddit, hate or not hate.
    #used to tell html code what color output link should be.
    classList = []

    if prediction:
        classList.append('hate')
    else:
        classList.append('nothate')

    #attach a class to all the subreddits
    for i,subreddit in enumerate(simSubredditList):
        simSubredditList[i] = "/r/{}".format(subreddit)

        if ishateful(subreddit):
            classList.append('hate')
        else:
            classList.append('nothate')


    return render_template('index2.html', tweet=tweet,subreddit=simSubredditList, classList=classList)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
