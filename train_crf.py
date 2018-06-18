from __future__ import print_function
import argparse
from dataUtils import convertTo4Tag, lineToStr, Indexer, textFileToDataset

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd
import elephas

from crfUtils import CRF
from keras.layers import Dense, Embedding, Conv1D, Lambda
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.initializers import glorot_normal

def run_train(master_name, filename, outname):
    import pyspark
    conf = pyspark.SparkConf().setAppName("CRF").setMaster(master_name)
    sc = pyspark.SparkContext(conf=conf)
    tfile = sc.textFile(filename)
    dataset = textFileToDataset(tfile)
    Indexer.prepareIndexer(dataset)

    dataset = dataset.collect()
    print('[Prepare Trainloader]')
    trainloader = Indexer.convertToBatchIter(dataset, 10000)
    embedding_size = 128
    print('[Char account] {}'.format(len(Indexer._chars)))

    crf_model = CRF(True, name='CRF')
    cnn_model = Sequential([
        Embedding(len(Indexer._chars)+1, embedding_size),
        Conv1D(128, 3, activation='relu', padding='same',\
               kernel_constraint=maxnorm(1.0), name='conv1'),
        Conv1D(128, 3, activation='relu', padding='same',\
               kernel_constraint=maxnorm(1.0), name='conv2'),
        Dense(5),
        Lambda(lambda x:x)
        #crf_model
    ])
    '''
    embed=Embedding(len(Indexer._chars)+1, embedding_size)(inph)
    cnn=Conv1D(128, 3, activation='relu', padding='same')(embed)
    cnn=Conv1D(128, 3, activation='relu', padding='same')(cnn)
    tag_score=Dense(5)(cnn)
    '''
    crf_model.trans = cnn_model.layers[-1].add_weight(name='transM', shape=(crf_model.num_labels, crf_model.num_labels), initializer=glorot_normal())
    cnn_model.compile(loss=crf_model.loss,
                optimizer='adam',
                metrics=[crf_model.accuracy]
                )
    cnn_model.summary()
    print(cnn_model.weights)
    print(crf_model.trans.__repr__())
    optimizerE = elephas.optimizers.Adam()
    spark_model = SparkModel(sc, cnn_model, optimizer=optimizerE,\
                    frequency='epoch', mode='synchronous', num_workers=2,\
                             ) #custom_objects={'CRF': crf_model})

    ep = 0
    while ep<10:
        X,Y = next(trainloader)
        ep = ep+len(Y)
        rdd = to_simple_rdd(sc, X, Y)
        model = spark_model.master_network
        model.save(outname+'_{}'.format(ep))
        spark_model.train(rdd, nb_epoch=5, batch_size=100, validation_split=0.1, verbose=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['convert', 'train'])
    parser.add_argument('filename', help='file(directory) to convert or formated file to train')
    parser.add_argument('outname', help='output file prefix')
    parser.add_argument('--master', help='master ip:port or yarn', default='local')
    args = parser.parse_args()

    if args.phase == 'convert':
        outname = args.outname
        print("[main] Convert {} to {}".format(args.filename, outname))
        tagrdd = convertTo4Tag(args.filename, args.master)
        strrdd = tagrdd.map(lineToStr) # rdd
        strrdd.saveAsTextFile(outname)
        print('[main] Finished')
    else:
        print("[main] Train CRF model")
        model = run_train(args.master, args.filename, args.outname)
