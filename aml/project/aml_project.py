#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cornell Tech
AML Project Fall 2021

This script ingests RF data in .npy format, and 
trains/tests either an SVM, random forest, or CNN model

TODO:
    - Finish CNN and random forests
    - Move formatIQTimeData into main function, no need to have it in each model
    - Make formatIQTimeData able to ingest data with various sample rates and
      resample (either upssample to downsample) to a common sample rate

Date Created:
11/1/21

Authors:
Rich Seibert
Larry Xu
"""


import sys
# sdr imports
from pylab import *
from rtlsdr import *
# math and plotting
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fftshift
import numpy as np
# preprocessing
from sklearn import preprocessing
# svm
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
# cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical



def formatIQTimeData(fileNames):
    # take in raw IQ samples from pred recording and preprocess
    # by doing mag, normalizing to 0-1, and then cutting it up
    data = []
    targets = []
    for fileName in fileNames:
        # read in time domain data
        dataPart = np.load(fileName)
        # TODO just using mag data, try using phase as well
        # convert to mag normalized
        dataPart = np.abs(dataPart)
        dataPart = dataPart / np.max(dataPart)
        # cut out first part where sdr hasn't settled
        dataPart = dataPart[1024*5:]
        # TODO try spectrogram/fft here and use freq data to train the model
        # reshape into 2d array
        wrapSize = 1024
        for i in range(0, len(dataPart), wrapSize):
            data.append(dataPart[i:i+wrapSize])
        # label is first few characters on file that indicates modulation type
        fileNameStart = fileName.rfind('/')+1
        print(fileName)
        label = fileName[fileNameStart:fileName.find('_')]
        rowsAdded = int(len(dataPart)/wrapSize)
        targets = np.concatenate((targets, np.array([label]*rowsAdded)))
    return np.array(data), targets


def plot(samples, time=False, phase=False, spec=True):
    samples = np.load(fileName)
    if time:
        # time domain data mag
        plt.plot(np.abs(samples))
        plt.title('Time domain data')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.show()
    if phase:
        # time domain data phase
        plt.plot(np.angle(samples))
        plt.title('Time domain data')
        plt.xlabel("Time (s)")
        plt.ylabel("Phase")
        plt.show()
    if spec:
        # Matplotlib.pyplot.specgram()
        plt.specgram(samples, Fs=1e6, cmap="rainbow")
        plt.title('Spectrogram')
        plt.xlabel("Time (s)")
        plt.ylabel("IF (MHz)")
        plt.show()
        '''
        # scipy spectrogram
        f, t, Sxx = signal.spectrogram(samples, fs)
        dataSpectrogram = fftshift(Sxx, axes=0)
        specdb = 10*np.log10(dataSpectrogram)
        plt.pcolormesh(t, fftshift(f), specdb, shading='gouraud')
        plt.xlabel('Time (s)')
        plt.ylabel('IF (MHz)')
        plt.show()
        '''


def CNN(fileNames):
    # reformat data recorded by sdr into 2d array where each row is 
    # one array of one type of signal data. If multiple files
    # are input it will concatenate all of them together
    data, targets = formatIQTimeData(fileNames)

    # label encoding on modulation types which are strings
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    le = preprocessing.LabelEncoder()
    le.fit(targets)
    labels = le.classes_
    print("Running classification for the following signals:", labels)
    targets = le.transform(targets)

    reshaped = []
    for i in range(data.shape[0]):
        reshaped.append(data[i].reshape(32, 32))
    data = np.array(reshaped)
    
    # Split data into 50% train and 50% test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.5, shuffle=True)
    
    # convert to OHE
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # define cnn using Sequential
    model = Sequential()
    # Convolution layer
    model.add (
        Conv2D(32, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (32 , 32 , 1 ))
    )
    # 2nd Convolution layer
    model.add (
        Conv2D(64, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (32 , 32 , 1 ))
    )
    # Maxpooling layer
    model.add(MaxPooling2D((2, 2)))
    # 2nd Maxpooling layer
    model.add(MaxPooling2D((2, 2)))
    # Flatten output
    model.add(Flatten())
    # Dropout later
    model.add(Dropout(0.5))
    # Dense layerof 100 neurons
    model.add(
        Dense(100,
        activation = 'relu',
        kernel_initializer = 'he_uniform')
    )
    model.add(
        Dense(len(labels),
        activation = 'softmax'))
    # initialize optimizer
    opt = SGD(learning_rate=0.01, momentum=0.9)
    # compile model
    model.compile(
        optimizer=opt,
        loss = 'categorical_crossentropy',
        metrics = ['accuracy']
    )

    model.summary()

    # run for 10 epochs
    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=32,
        epochs=1,
        validation_data=(x_test, y_test)
    )
    
    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    results = model.evaluate(x_test, y_test, verbose=2)

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def KNN(fileNames):
    print("Not implemented yet")
    return


def randomForest(fileNames):
    print("Not implemented yet")
    return


def SVM(fileNames):
    # reformat data recorded by sdr into 2d array where each row is 
    # one array of one type of signal data. If multiple files
    # are input it will concatenate all of them together
    data, targets = formatIQTimeData(fileNames)
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.5, shuffle=True)
    
    # Create a classifier: a support vector classifier
    clf = svm.SVC(gamma=0.001)
    
    # Learn the digits on the train subset
    clf.fit(X_train, y_train)
    
    # Predict the value of the digit on the test subset
    predicted = clf.predict(X_test)
    
    print(
        f"Classification report for classifier {clf}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n")


def record(writeData, modId, num1k, centerFreq, plot):
    writeData = int(writeData)
    num1k = int(num1k)
    plot = int(plot)
    # capture signal from rtl sdr
    sdr = RtlSdr()
    # 200 KHz for FM, 20 for AM
    fs = 1e6
    sdr.sample_rate = fs
    sdr.center_freq = float(centerFreq)*1e6
    sdr.gain = 'auto'
    
    samples = sdr.read_samples(num1k*1024)
    sdr.close()
    
    fileName =  modId + '_' + centerFreq + '_' + string_fs + \
                '_' + 'raw_data.npy', samples
    
    if writeData:
        print("Writting Data")
        # write raw iq data to modify later if needed
        string_fs = str(fs/1e6)+'MHz'
        np.save(fileName)
        
        # don't save off spectrogram data anymore, no reason to...
        # can just make spectrogram from raw samples
        '''
        # write normalized spectrogram mag data
        f, t, Sxx = signal.spectrogram(samples, fs) #nfft=1024, defaults to 256
        dataSpectrogram = fftshift(Sxx, axes=0)
        # normalize to 0-1
        dataSpectrogram = dataSpectrogram / np.max(dataSpectrogram)
        #dataSpectrogram = 10*np.log10(dataSpectrogram)
        np.save(modId + '_' + centerFreq + '_' + string_fs + '_' + 
               'norm_spec_data.npy', dataSpectrogram)
        '''
    
    if plot:
        plot(samples)


if __name__ == "__main__":
    path = "/home/rich/radio_stuff/aml_project_recordings/sdr_recordings/"
    fileNames = ["gsm_852.038_1.0MHz_raw_data.npy",
                 "unk_454.835_1.0MHz_raw_data.npy",
                 "fm_100.3_1.0MHz_raw_data.npy",
                 "unk2_905.602_1.0MHz_raw_data.npy",
                 "nbfm_162.55_1.0MHz_raw_data.npy",
                 "unk_454.944_1.0MHz_raw_data.npy",
                 "fm_103.5_1.0MHz_raw_data.npy",
                 "unk2_912_1.0MHz_raw_data.npy"]
    fileNames = [path+x for x in fileNames]
    
    if len(sys.argv) <= 1:
        print("How to use this script:\n")
        print("Example: " +
              "python3.9 aml_project.py cnn fm_100.3_1.0MHz_raw_data.npy" +
              "fm_103.5_1.0MHz_raw_data.npy gsm_852.038_1.0MHz_raw_data.npy \n")
        print("--svm = train support vector machine model and test")
        print("    file names")
        print("--getdata = record raw iq data from SDR")
        print("    writeData  - write raw iq and spectrogram data")
        print("    modulation - few character mod identifier")
        print("    num1k      - number of 1k samples to take")
        print("    centerFreq - center frequency in MHz")
        print("    plot       - show plots of just taken data")
        print("--plot = plots data from raw IQ files")
        print("    file names")
    elif sys.argv[1] == 'svm':
        SVM(fileNames)
    elif sys.argv[1] == 'cnn':
        CNN(fileNames)
    elif sys.argv[1] == 'getdata':
        record(*sys.argv[2:])
    elif sys.argv[1] == 'plot':
        fileNames = sys.argv[2:]
        for fileName in fileNames:
            samples = np.load(fileName)
            plot(samples)
