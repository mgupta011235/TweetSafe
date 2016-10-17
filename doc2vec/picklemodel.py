import gensim
import cPickle as pickle

if __name__ == '__main__':
    '''This script tests a doc2vec model on the training,cross val and test set'''

    print "starting..."

    #model paths
    modelPath = '../../doc2vec_models/basemodel2/basemodel2.doc2vec'

    print "loading model..."
    model = gensim.models.Doc2Vec.load(modelPath)

    print "pickling..."
    pickle.dump(model, open('../../data/basemodel2.p', 'wb'))
