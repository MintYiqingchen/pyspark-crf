from __future__ import print_function
import pyspark
from operator import add
import numpy as np
class Indexer:
    _tagdict = {'B':0, 'E':1,'M':2, 'S':3, 'N':4}
    _chars = {}
    _id2char = {}
    _char2id = {}
    _id2tag = {}
    @staticmethod
    @property
    def tagdict():
        return Indexer._tagdict

    @staticmethod
    def prepareIndexer(corps, min_count=20):
        '''@corps: RDD [[(char, label)]]'''
        tmp=corps.map(lambda x:x+[('\0','N')]).flatMap(lambda x:x)
        tmp = tmp.map(lambda x:(x[0], 1)).reduceByKey(add)
        tmp = tmp.filter(lambda x: x[1]>=min_count).collect()
        Indexer._chars = tmp
        Indexer._id2char = {i+1:j for i,j in enumerate(Indexer._chars)}
        Indexer._char2id = {j:i for i,j in Indexer._id2char.items()}
        Indexer._id2tag = {j:i for i,j in Indexer._tagdict.items()}
        print('[Indexer] Finish Prepare')

    @staticmethod
    def convertToBatchIter(tupList, batch_size, shuffle=True):
        '''tupList:[[(char, label)]]'''
        def train_generator(tupList):
            if shuffle:
                np.random.shuffle(tupList)
            now_index = 0
            char2id = Indexer._char2id
            tag2id = Indexer._tagdict
            while True:
                if now_index + batch_size <= len(tupList):
                    next_index = now_index+batch_size
                    A = [tupList[i] for i in range(now_index, next_index)]
                else:
                    next_index = batch_size - len(tupList) + now_index
                    A = tupList[now_index:] + tupList[: next_index]
                now_index = next_index

                maxlen = max([len(x) for x in A])
                print(A[0])
                X = [list(map(lambda x:x[0],sent)) for sent in A]
                Y = [list(map(lambda x:x[1],sent)) for sent in A]
                X = [x+[0]*(maxlen-len(x)) for x in X]
                Y = [y+[4]*(maxlen-len(y)) for y in Y]
                yield X,Y
        return train_generator(tupList)
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
    '''@tfile: textFile RDD'''
    def func(line):
        l = line.strip().split()
        res = list(map(lambda tk: tuple(tk.split('/')), l))
        return res
    char_tag = tfile.map(func)
    return char_tag
