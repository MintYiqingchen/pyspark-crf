# pyspark-crf
conditional random field implement by pyspark
## Get Start
1. Prepare execution environment
```
$ wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda2-5.2.0-Linux-x86_64.sh
$ source activate
$ conda create -n spark --copy -y python=2 numpy scipy pandas tensorflow keras=2.1.6 elephas
$ conda install mkl-service
$ export MKL_THEAREADING_LAYER=GNU
```
If other independency is reported, please install them as well. 

2. Prepare [Hadoop 2.9.1](http://hadoop.apache.org/) and [Spark 2.3.1](http://spark.apache.org/)

3. Run train_crf.py to get input data
```
$ python train_crf.py -h
$ python train_crf.py --master <yarn or local> convert <input file(dir)> <output file>
``` 
For example, run in local
```
$ python train_crf.py convert file:///home/share/train file:///home/share/output
$ python train_crf.py train file:///home/share/output file:///home/share/a
```
If you want to run it on yarn, zip the python environment:
```
$ cd /root/anaconda2/envs/spark -r spark.zip spark
$ PYSPARK_PYTHON=./ANACONDA/mlpy_env/bin/python spark-submit \ 
    --conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./ANACONDA/mlpy_env/bin/python 
    --master yarn
    --archives /tmp/mlpy_env.zip
```
