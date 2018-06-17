from __future__ import print_function
import argparse
import os
from dataUtils import convertTo4Tag, lineToStr, Indexer, textFileToDataset

from keras.layers.convolutional import Convolution2D, MaxPooling2D
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from keras.utils import to_categorical
from crfUtils import CRF
from keras.layers import Dense, Embedding, Conv1D, Input
from keras.models import Sequential, Model
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
    trainloader = Indexer.convertToBatchIter(dataset[:5000], 1)
    embedding_size = 128
    inph = Input(shape=(None,), dtype='int32')
    cnn_model = Sequential([
        Embedding(len(Indexer._chars)+1,embedding_size),
        Conv1D(128, 3, activation='relu', padding='same'),
        Conv1D(128, 3, activation='relu', padding='same'),
        Dense(5)
    ])

    tag_score = cnn_model(inph)
    crf_model = CRF(True)
    tag_score = crf_model(tag_score)
    model = Model(inputs=inph, outputs=tag_score)
    model.summary()

    model.compile(loss=crf_model.loss,
                optimizer='adam',
                metrics=[crf_model.accuracy]
                )

    spark_model = SparkModel(sc,model)
    ep = 0
    while ep<1000:
        X,Y = next(trainloader)
        rdd = to_simple_rdd(sc, X, Y)
        spark_model.train(rdd, nb_poch=1, batchsize=len(X), verbose=1)
        ep += 1
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
