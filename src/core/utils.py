#!/usr/bin/env python2
# -*- coding: utf-8 -*

import pylab
import matplotlib.pyplot as plt

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise

def save_fig(output_dir,filename):
	mkdir_p(output_dir)
	pylab.savefig(filename)
	plt.close()

#split a list of tuples into two different lists
def split_tuples(tuple_list):

    unzip=map(list, zip(*tuple_list))						#split into two arrays
    list1=unzip[0]										    #first list
    list2=unzip[1]								            #second list

    return list1,list2