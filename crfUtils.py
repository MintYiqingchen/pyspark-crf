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

# -*- coding:utf-8 -*-

from keras.layers import Layer
import keras.backend as K

class CRF(Layer):
    def __init__(self, ignore_last_label=False, **kwargs):
        self.ignore_last_label = 1 if ignore_last_label else 0
        super(CRF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        output = K.logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True) # 逐标签得分
        labels1 = K.expand_dims(labels[:, :-1], 3)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score
    def call(self, inputs):
        return inputs
    def loss(self, y_true, y_pred):
        mask = 1-y_true[:,1:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        init_states = [y_pred[:,0]]
        log_norm,_,_ = K.rnn(self.log_norm_step, y_pred[:,1:], init_states, mask=mask) # 计算Z向量（对数）
        log_norm = K.logsumexp(log_norm, 1, keepdims=True) # 计算Z（对数）
        path_score = self.path_score(y_pred, y_true) # 计算分子（对数）
        return log_norm - path_score # 即log(分子/分母)
    def accuracy(self, y_true, y_pred): # 训练过程中显示逐帧准确率的函数，排除了mask的影响
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)
