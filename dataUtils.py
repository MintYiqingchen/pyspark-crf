from __future__ import print_function
from operator import add
import numpy as np
import pyspark


class Indexer(object):
    '''a class to record tag and chars'''
    def __init__(self):
        self.tagdict = {'B': 0, 'E': 1, 'M': 2, 'S': 3, 'N': 4}
        self.chars = {}
        self.id2char = {}
        self.char2id = {}
        self.id2tag = {}

    def prepareIndexer(self,corps, min_count=2):
        '''@corps: RDD [[(char, label)]]'''
        tmp=corps.map(lambda x:x+[('\0','N')]).flatMap(lambda x:x)
        tmp = tmp.map(lambda x:(x[0], 1)).reduceByKey(add)
        tmp = tmp.filter(lambda x: x[1]>=min_count).collect()
        self.chars = dict(tmp)
        self.id2char = {i+1:j for i,j in enumerate(self.chars)}
        self.char2id = {j:i for i,j in self.id2char.items()}
        self.id2tag = {j:i for i,j in self.tagdict.items()}
        assert len(self.char2id)>0, 'Indexer.char2id is empty'
        print('[Indexer] Finish Prepare')
    def convertToElephasFormat(self, raw_rdd, shuffle=True):
        '''[[(Char, tag)]] => [[(int, onehot)]]'''
        rdd = raw_rdd.mapPartitions(self.convertToBatch)
        if shuffle:
            rdd = rdd.repartition(rdd.getNumPartitions())
        return rdd

    def convertToBatch(self,tupList):
        '''tupList:[[(char, label)]]'''
        char2id = self.char2id
        assert len(char2id)>0, 'char2id is empty'
        tag2id = self.tagdict
        A = list(tupList)

        maxlen = max([len(x) for x in A])
        X = [list(map(lambda x:char2id.get(x[0],0),sent)) for sent in A]
        Y = [list(map(lambda x:tag2id.get(x[1],0),sent)) for sent in A]
        X = [x+[0]*(maxlen-len(x)) for x in X]
        Y = [y+[4]*(maxlen-len(y)) for y in Y]
        X,Y = np.array(X) , to_categorical(Y,5)
        assert np.all(np.sum(X,axis=1)), X
        return list(zip(X,Y))

def to_categorical(data, num_classes=None):
    y = np.array(data, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def tagForSentence(tokens):
    '''@tokens: List[token]
    return: @[Char,Tag]
    '''
    res = []
    for token in tokens:
        if len(token)==1:
            tag = ['S']
        else:
            tag = ['B']+['M']*(len(token)-2)+['E']
        res.extend([(c,t) for c,t in zip(token, tag)])
    return res

def lineToStr(line):
    '''@line: [(a,b)]
    return: 'a/b a2/b2 ...'
    '''
    res = ' '.join(map(lambda t:unicode(t[0])+'/'+unicode(t[1]), line))
    return res
def convertTo4Tag(input_file,master_name):
    conf = pyspark.SparkConf().setAppName("4Tag").setMaster(master_name)
    sc = pyspark.SparkContext(conf=conf)
    lines = sc.textFile(input_file) #.persist() cache it for perfermance

    # for every token in each line
    oriLen = lines.count()
    lines = lines.map(lambda l: l.strip().split()).filter(lambda l: len(l)>0)
    filLen = lines.count()
    print("[Tag] Filter empty lines: {}->{}".format(oriLen, filLen))

    tags = lines.map(tagForSentence) # RDD: [[(Char, position)]]
    return tags

def textFileToDataset(tfile):
    '''@tfile: textFile RDD
    return: RDD [[(char, tag)]]
    '''
    def func(line):
        l = line.strip().split()
        res = list(map(lambda tk: tuple(tk.split('/')), l))
        return res
    char_tag = tfile.map(func)
    return char_tag
