import argparse
import os
from dataUtils import convertTo4Tag, lineToStr
from __future__ import print_function
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase',choices=['convert', 'train'])
    parser.add_argument('filename', help='file(directory) to convert or formated file to train')
    parser.add_argument('outname', help='output file prefix')
    parser.add_argument('--master', help='master ip:port or yarn', default='local')
    args = parser.parse_args()

    if args.phase=='convert':
        outname = args.outname
        print "[main] Convert {} to {}".format(args.filename, outname)
        tagrdd = convertTo4Tag(args.filename, args.master)
        strrdd=tagrdd.map(lineToStr) # rdd
        strrdd.saveAsTextFile(outname)
        print '[main] Finished'
    else:
        pass 