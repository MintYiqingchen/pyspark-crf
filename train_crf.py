from __future__ import print_function
import argparse
import os
from dataUtils import convertTo4Tag, lineToStr, Indexer, textFileToDataset

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from elephas.spark_model import SparkModel

from keras.utils import to_categorical
#from crfUtils import CRF
from keras.layers import Dense, Embedding, Conv1D, Input
import keras.backend as K

def run_train(master_name, filename):
    import pyspark
    conf = pyspark.SparkConf().setAppName("CRF").setMaster(master_name)
    sc = pyspark.SparkContext(conf=conf)
    tfile = sc.textFile(filename)
    dataset = textFileToDataset(tfile)
    Indexer.prepareIndexer(dataset)

    dataset = dataset.collect()
    print('[Prepare Trainloader]')
    print(dataset[0])
    trainloader = Indexer.convertToBatchIter(dataset[:5000], 1)
    embedding_size = 128
    inph = Input(shape=(None,), dtype='int32')
    print(next(trainloader))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['convert', 'train'])
    parser.add_argument('filename', help='file(directory) to convert or formated file to train')
    parser.add_argument('outname', help='output file prefix')
    parser.add_argument('--master', help='master ip:port or yarn', default='local')
    args = parser.parse_args()

    if args.phase=='convert':
        outname = args.outname
        print("[main] Convert {} to {}".format(args.filename, outname))
        tagrdd = convertTo4Tag(args.filename, args.master)
        strrdd=tagrdd.map(lineToStr) # rdd
        strrdd.saveAsTextFile(outname)
        print('[main] Finished')
    else:
        print("[main] Train CRF model")
        model = run_train(args.master, args.filename)
