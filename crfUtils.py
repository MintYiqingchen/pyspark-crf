import numpy as np
from pyspark.ml.feature import Word2Vec
from __future__ import print_function
def viterbi(pathWeight, initW):
    '''@pathWeight: step*kind_t*kind_(t+1) t=1..T-1
    return: @bpath: best path
            @prob: path probability
    '''
    accum_prob = initW
    bpath = []
    for i,tpoint in enumerate(pathWeight): # for every step
        tmpW = np.array(tpoint)
        # prob to next step
        prob = np.asarray(accum_prob) + tmpW.T # kind_(t+1)*kind_t
        from_t = np.argmax(prob, axis=1).T
        accum_prob = [p[t] for (t, p) in zip(from_t, prob)]
        bpath.append(list(from_t))
    return bpath, accum_prob

def trainW2V(df, vsize=16, **kwargs):
    '''@df: a dataframe ['c','c']'''
    word2v = Word2Vec()
    word2v.set
