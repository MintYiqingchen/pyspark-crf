from __future__ import print_function
import numpy as np
from keras.layers import Layer
import keras.backend as K
import tensorflow as tf

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

def logsumexp(x, axis=None, keepdims=False):
    return tf.reduce_logsumexp(x, axis, keepdims)

class CRF(object):
    def __init__(self, label_size=5, ignore_last_label=False, **kwargs):
        self.ignore_last_label = 1 if ignore_last_label else 0
        #super(CRF, self).__init__(**kwargs)
        self.num_labels = label_size - self.ignore_last_label

        # self.trans = self.get_transfer_matrix()
        # print(self.trans.__repr__())
        #self.trans=K.random_uniform_variable(shape=(self.num_labels, self.num_labels),
        # low=0, high=1)
    def get_transfer_matrix(self):
        '''generate a transfer matrix as shape (k,k)'''
        a = K.variable(K.random_normal((self.num_labels, self.num_labels), stddev=0.1))
        return a

    def build(self, input_shape):
        self.num_labels = input_shape[-1] - self.ignore_last_label
        # transfer matrix
        self.trans = self.add_weight(name='crf_trans',
                                     shape=(self.num_labels, self.num_labels),
                                     initializer='glorot_uniform',
                                     trainable=True)
    def log_norm_step(self, inputs, states):
        '''
        @inputs: is probability of every node on t-th step
        @states: accumulate probability of each node until t-th step
        '''
        states = K.expand_dims(states[0], 2) # (batch_size, output_dim, 1)
        trans = K.expand_dims(self.trans, 0) # (1, output_dim, output_dim)
        # calculate use broadcast
        # exp(states)*exp(trans)=exp(states+trans)
        # output is log(sum(exp(accum_prob+trans_prob), same_next_node_axis))
        output = logsumexp(states+trans, 1) # (batch_size, output_dim)
        return output+inputs, [output+inputs]
    def path_score(self, inputs, labels):
        '''calculate optimize objection'''
        # node score f(y_i, x)
        point_score = K.sum(K.sum(inputs*labels, 2), 1, keepdims=True)
        # (sample, label_(n-1) ,1, cls)
        labels1 = K.expand_dims(labels[:, :-1], 3)
        # (sample, 1, label_1, cls)
        labels2 = K.expand_dims(labels[:, 1:], 2)
        labels = labels1 * labels2
        # (1, 1, cls, cls)
        trans = K.expand_dims(K.expand_dims(self.trans, 0), 0)
        # g(y_i, y_(i+1), x)
        trans_score = K.sum(K.sum(trans*labels, [2,3]), 1, keepdims=True)
        return point_score+trans_score
    def call(self, inputs):
        return inputs
    def loss(self, y_true, y_pred):
        # (samples, time, 1): if a time step's value should be ignored,
        # its mask value should be 1
        mask = 1-y_true[:, 1:, -1] if self.ignore_last_label else None
        # drop the final label 'N'
        y_true, y_pred = y_true[:, :, :self.num_labels], y_pred[:, :, :self.num_labels]
        init_states = [y_pred[:, 0]]
        # recurrent calculate probability from step 0
        log_norm, _, _ = K.rnn(self.log_norm_step, y_pred[:, 1:], init_states, mask=mask)
        # Z(x): log(prob) of accumulating probability of each final node
        log_norm = logsumexp(log_norm, 1, keepdims=True)
        path_score = self.path_score(y_pred, y_true)
        return log_norm - path_score # log(Z)-pathscore

    def accuracy(self, y_true, y_pred):
        mask = 1-y_true[:,:,-1] if self.ignore_last_label else None
        y_true,y_pred = y_true[:,:,:self.num_labels],y_pred[:,:,:self.num_labels]
        isequal = K.equal(K.argmax(y_true, 2), K.argmax(y_pred, 2))
        isequal = K.cast(isequal, 'float32')
        if mask == None:
            return K.mean(isequal)
        else:
            return K.sum(isequal*mask) / K.sum(mask)
