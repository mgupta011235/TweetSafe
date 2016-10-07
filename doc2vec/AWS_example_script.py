# Install libraries
!sudo pip install gensim==0.10.3

!sudo pip install nltk

!sudo pip install python-cjson

import nltk
nltk.download('all-corpora')

# WORD CLEAN AND TOKENIZE FUNCTION
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

stopwords = stopwords.words('english')
stopwords.append('[deleted]')
tokenizer = RegexpTokenizer(r'\w+')

def text_cleaner(text):
    '''
    INPUT: string of body text
    OUTPUT: List of tokenized lower case words with stopwords removed
    '''
    # Output tokenizes text and removes any stopwords and then outptus lowercased words
    return [word.lower() for word in tokenizer.tokenize(text) if not word.lower() in stopwords]



# Function to generate labeled sentences straight from the file
import cjson
from gensim.models import Doc2Vec
from gensim.models.doc2vec import LabeledSentence
import gensim.models.doc2vec


def reddit_comment_gen(pathway):
    '''
	INPUT: Pathway to database and num of comments to be generated. If everything is True, all comments returned.
	OUTPUT: Generator label and tokenized comment

	'''

    ## Generate all labeled sentences from file

        # Iterate through N JSON objects in file
    with open(pathway) as myfile:
        for item in myfile:


            # put in try statement here

            # Load the JSON object
            json_object = cjson.decode(item)

            # Clean and tokenize text
            body = text_cleaner(json_object['body'])

            # generate
            yield LabeledSentence(body,labels=[str(json_object['subreddit'])])



# Function that builds the models

import multiprocessing
import numpy as np

def build_model(pathway):

    cores = multiprocessing.cpu_count()
    assert gensim.models.doc2vec.FAST_VERSION > -1


    d2v_reddit_model = Doc2Vec( dm=0, size=100, negative=5, hs=0, min_count=2, workers=cores)
    d2v_reddit_model.train_words = False

    d2v_reddit_model.build_vocab(reddit_comment_gen(pathway)) #sentence_gen(reddit_data))


    #d2v_reddit_model.train(reddit_comment_gen(pathway,10000)) #sentence_gen(reddit_data))

    for epoch in range(10):

        d2v_reddit_model.train(reddit_comment_gen(pathway))
        d2v_reddit_model.alpha -= 0.002  # decrease the learning rate
        d2v_reddit_model.min_alpha = d2v_reddit_model.alpha  # fix the learning rate, no decay

    return d2v_reddit_model
