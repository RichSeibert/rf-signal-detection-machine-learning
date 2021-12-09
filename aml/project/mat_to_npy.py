#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Convert .mat (matlab data files) into .npy files

Date Created:
12/7/21

Authors:
Rich Seibert
"""

import sys
import scipy.io
import numpy as np
import glob

def mat_to_npy(fileName, verbose=False):
    if verbose:
        print("Converting", fileName)
    matFile = scipy.io.loadmat(fileName)
    # TODO wtf is going on here
    print(matFile['data'][0][0][0])
    np.save(fileName[:-4] + '.npy', matFile)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("To run this script stand-alone, input a .mat file" + 
              " or directory that contains .mat files")
    else:
        verbose = True
        if sys.argv[1][-4:] == '.mat':
            mat_to_npy(sys.argv[1], verbose)
        else:
            files = glob.glob(sys.argv[1] + "/*.mat")
            for file in files:
                mat_to_npy(file, verbose)
