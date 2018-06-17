import pyspark
from __future__ import print_function
class Indexer:
    _tagdict = {'B':0, 'E':1,'M':2, 'S':3}
    @staticmethod
    @property
    def tagdict():
        return _tagdict
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
    print "[Tag] Filter empty lines: {}->{}".format(oriLen, filLen)
    
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

def prepareW2VDataFrame(corps):
    '''@corps: RDD [[(char, label)]]
    return: RDD [(char, label)]'''
    res=corps.map(lambda x:x+[('\0','4')]).flatMap(lambda x:x)
    return res