import pyspark
class Indexer:
    _tagdict = {0:'B', 1:'E',2:'M', 3:'S'}
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
            tag = [3]
        else:
            tag = [0]+[2]*(len(token)-2)+[1]
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