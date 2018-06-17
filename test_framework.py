from __future__ import absolute_import
from __future__ import print_function
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils

from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

from pyspark import SparkContext, SparkConf

APP_NAME = "mnist"
MASTER_IP = 'local[24]'

# Define basic parameters
batch_size = 128
nb_classes = 10
nb_epoch = 5

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

X_train = X_train.astype("float32")
X_test = X_test.astype("float32")
X_train /= 255
X_test /= 255


print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                                                border_mode='valid',
                                                input_shape=(1, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')

## spark
conf = SparkConf().setAppName(APP_NAME).setMaster(MASTER_IP)
sc = SparkContext(conf=conf)

# Build RDD from numpy features and labels
rdd = to_simple_rdd(sc, X_train, Y_train)

# Initialize SparkModel from Keras model and Spark context
spark_model = SparkModel(sc,model)

# Train Spark model
spark_model.train(rdd, nb_epoch=nb_epoch, batch_size=batch_size, verbose=0, validation_split=0.1, num_workers=24)

# Evaluate Spark model by evaluating the underlying model
score = spark_model.get_network().evaluate(X_test, Y_test, show_accuracy=True, verbose=2)
print('Test accuracy:', score[1])
