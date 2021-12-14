#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Cornell Tech
AML Project Fall 2021

This script ingests RF data in .npy format, and 
trains/tests either an SVM, random forest, or CNN model

TODO:
        - show performance of SVN with small, then larger data set, try adjusting
          inputs like size from 1024 to 4096, then try feeding in phase data as well.
          Make plots and print outs of performance.
          Then move onto CNN, do same thing
    - Finish CNN and random forests
    - start with simple model, train/dev/test, print metrics (lecture 20 p2/3)
      and plots, evalutate whats going on (overfitting, underfitting) and
      then try different things and improve model. Then move on to more
      advanced models like CNN. Maybe just try to test out svm and random
      forests some more and then primarily focus on CNN
        Need to keep tweeking and looking at dev results, then after that looks
        good, run once on test set and show results

Date Created:
11/1/21

Authors:
Rich Seibert
Larry Xu
"""


import sys
import glob
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
# metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
# svm, random forests, and knn
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# cnn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers



def formatIQTimeData(fileNames, cutAmount=0):
    # take in raw IQ samples from pred recording and preprocess
    # by doing mag, normalizing to 0-1, and then cutting it up
    data = []
    targets = []
    for fileName in fileNames:
        # read in time domain data
        dataPart = np.load(fileName)
        # convert to mag normalized
        dataPartMag = np.abs(dataPart)
        dataPartMag = dataPartMag / np.max(dataPartMag)
        # convert to phase normalized
        dataPartPhase = np.angle(dataPart)
        dataPartPhase = dataPartPhase/ np.max(dataPartPhase)
        # cut out first part where sdr hasn't settled
        dataPartMag = dataPartMag[1024*5:]
        dataPartPhase = dataPartPhase[1024*5:]
        # TODO try spectrogram/fft here and use freq data to train the model
        # reshape into 2d array
        wrapSize = 4096
        rowsAdded = 0
        for i in range(0, len(dataPartMag)-wrapSize, wrapSize):
            rowsAdded += 1
            data.append(dataPartMag[i:i+wrapSize])
            # TODO just using mag data, try using phase as well
            # CNN seems to be able to accept complex data, but I can't normalize it
            #data.append(dataPartPhase[i:i+wrapSize])
        # label is first few characters on file that indicates modulation type
        fileNameStart = fileName.rfind('/')+1
        # cut the entire file path out, just keep file name
        fileNameOnly = fileName[fileNameStart:]
        label = fileNameOnly[:fileNameOnly.find('_')]
        targets = np.concatenate((targets, np.array([label]*rowsAdded)))
    # optional cut out data (used for when you want to make training faster)
    if cutAmount > 1:
        data = data[::cutAmount]
        targets = targets[::cutAmount]

    return np.array(data), targets


def plot(fileName, time=True, phase=False, spec=True):
    samples = np.load(fileName)
    print("++++++ NOTE: fs = 1e6 +++++++")
    print("Number of 1k samples=", samples.shape[0]/1024)
    print("Data type =", type(samples[0]))
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


def getMetrics(model, y_test, predicted, labels, plot=False, savePlot=False):
    # matrics report with f1 score, recall, precision
    print(
        f"Classification report for classifier {model}:\n"
        f"{metrics.classification_report(y_test, predicted)}\n")
    # f1 score
    f1Score = metrics.f1_score(y_test, predicted, average='micro')
    if plot:
        # confusion matrix
        cm = confusion_matrix(y_test, predicted, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=labels)

        disp.plot()
        plt.title(str(model) + " : F1 Score= " + "%.3f" % f1Score)
        plt.show()
    
    return f1Score


def CNN(fileNames):
    # reformat data recorded by sdr into 2d array where each row is 
    # one array of one type of signal data. If multiple files
    # are input it will concatenate all of them together
    cut = 1
    data, targets = formatIQTimeData(fileNames, cut)

    # label encoding on modulation types which are strings
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
    le = preprocessing.LabelEncoder()
    le.fit(targets)
    labels = le.classes_
    print("Training and testing CNN model classification for the following signals:", labels)
    targets = le.transform(targets)

    reshaped = []
    # this is the length of on row
    dataCutSize = data[0].shape[0]
    sqSize = int(dataCutSize**0.5)
    for i in range(data.shape[0]):
        if data[i].shape[0] == dataCutSize:
            reshaped.append(data[i].reshape(sqSize, sqSize))
    data = np.array(reshaped)
    
    # Split data into 50% train and 50% test subsets
    x_train, x_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.1, shuffle=True, random_state=42)
    y_test_orig = y_test
    
    # convert to OHE
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # define cnn using Sequential
    model = Sequential()
    # Convolution layer
    model.add (
        Conv2D(sqSize, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (sqSize , sqSize , 1 ))
    )
    # 2nd Convolution layer
    model.add (
        Conv2D(sqSize*2, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (sqSize , sqSize, 1 ))
    )
    # Maxpooling layer
    model.add(MaxPooling2D((2, 2)))
    # 2nd Maxpooling layer
    model.add(MaxPooling2D((2, 2)))

    # TODO add later
    '''
    # Convolution layer
    model.add (
        Conv2D(sqSize, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (sqSize , sqSize , 1 ))
    )
    # 2nd Convolution layer
    model.add (
        Conv2D(sqSize*2, (3, 3),
        activation = 'relu', kernel_initializer = 'he_uniform' ,
        input_shape = (sqSize , sqSize, 1 ))
    )
    # Maxpooling layer
    model.add(MaxPooling2D((2, 2)))
    # 2nd Maxpooling layer
    model.add(MaxPooling2D((2, 2)))
    '''

    # Flatten output
    model.add(Flatten())
    # Dropout layer
    model.add(Dropout(0.5))
    # Dense layerof 100 neurons
    model.add(
        Dense(100,
        kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
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

    # run 
    print("Fit model on training data")
    history = model.fit(
        x_train,
        y_train,
        batch_size=8,
        epochs=10,
        validation_data=(x_test, y_test)
    )
    
    # Evaluate the model on the test data using `evaluate`
    #print("Evaluate on test data")
    #predicted = model.evaluate(x_test, y_test, verbose=2)

    # metrics, using sklearn instead of keras, have to re-predict the data again
    # predict crisp classes for test set
    predicted = model.predict(x_test) 
    predicted = np.argmax(predicted, axis=1)
    # remap to label strings
    predicted = le.inverse_transform(predicted)
    # remap y_test to label string
    y_test_orig = le.inverse_transform(y_test_orig)

    f1Score = getMetrics(model, y_test_orig, predicted, labels, True, True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.title("CNN: Epoch vs Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()


def KNN(fileNames):
    cut = 5
    data, targets = formatIQTimeData(fileNames, cut)
    train_x, test_x, train_y, test_y = train_test_split(data, 
                                                        targets, 
                                                        test_size=0.3, 
                                                        shuffle=True, 
                                                        random_state=42)

    f1_scores = []
    neighbors = []
    for n in range(1, 30):
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(train_x, train_y)
        predicted = clf.predict(test_x)
        f1 = metrics.f1_score(test_y, predicted, average='micro')
        
        neighbors.append(n)
        f1_scores.append(f1)
        
    plt.xlabel("N neighbors")
    plt.ylabel("F1 score")
    plt.title("KNN: N neighbors vs. f1 score")
    plt.plot(neighbors, f1_scores)
    plt.show()

    # make confusion matrix
    f1Score = getMetrics(clf, test_y, predicted, clf.classes_, True, True)


def randomForest(fileNames):
    cut = 5
    data, targets = formatIQTimeData(fileNames, cut)
    train_x, test_x, train_y, test_y = train_test_split(data, 
                                                        targets, 
                                                        test_size=0.3, 
                                                        shuffle=True, 
                                                        random_state=42)

    # fix num_tree = 1, experiment with depth
    f1_scores = []
    depths = []
    for d in range(1, 30):
        clf = RandomForestClassifier(n_estimators=1, max_depth=d).fit(train_x, train_y)
        pred_y = clf.predict(test_x)
        f1 = metrics.f1_score(test_y, pred_y, average='micro')
        depths.append(d)
        f1_scores.append(f1)
    
    plt.xlabel("depth")
    plt.ylabel("F1 score")
    plt.title("Random Forests: depths vs. F1 score")
    plt.plot(depths, f1_scores)
    plt.show()
    
    # fix depth = 7, experiment with num_tree
    f1_scores = []
    num_trees = []
    for n in range(1, 30):
        clf = RandomForestClassifier(n_estimators=n, max_depth=7).fit(train_x, train_y)
        predicted = clf.predict(test_x)
        f1 = metrics.f1_score(test_y, predicted, average='micro')
        num_trees.append(n)
        f1_scores.append(f1)
    
    plt.xlabel("N trees")
    plt.ylabel("F1 score")
    plt.title("Random Forests: N trees vs. F1 score")
    plt.plot(num_trees, f1_scores)
    plt.show()
    
    # run again with the best parameter
    #clf = RandomForestClassifier(n_estimators=30, max_depth=10).fit(train_x, train_y)
    clf = RandomForestClassifier().fit(train_x, train_y)
    predicted = clf.predict(test_x)
    f1 = metrics.f1_score(test_y, predicted, average='micro')

    # make confusion matrix
    f1Score = getMetrics(clf, test_y, predicted, clf.classes_, True, True)


def SVM(fileNames):
    # reformat data recorded by sdr into 2d array where each row is 
    # one array of one type of signal data. If multiple files
    # are input it will concatenate all of them together
    cut = 10
    data, targets = formatIQTimeData(fileNames, cut)

    print("Training and testing SVM model classification" + 
          " for the following signals:\n", set(targets))
    
    # Split data into 50% train and 50% test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        data, targets, test_size=0.3, shuffle=True, random_state=30)

    def runSVMs(reg=4, kern='rbf'):
        # Create a classifier: a support vector classifier
        clf = svm.SVC(C=reg, kernel=kern)
        # Learn the digits on the train subset
        clf.fit(X_train, y_train)
        # Predict the value of the digit on the test subset
        predicted = clf.predict(X_test)
        labels = clf.classes_
        model = clf
        f1 = metrics.f1_score(y_test, predicted, average='micro')
        return f1

    # try out different parameters
    #   C: regularization paramter, must be posistive, default = 1
    #   kernel: linear, poly, rbf, sigmoid, default = rbf
    #   degree (for poly only) default = 3
    #   gamma: kernel coefficient for rbf, poly, and sigmoid 
    #          (scale, auto, or float), default = scale
    regularization = [i**2 for i in range(1, 5)]
    regF1Scores = []
    kernel = ['linear', 'poly', 'rbf', 'sigmoid']
    kernelF1Scores= []
    for val in regularization:
        regF1Scores.append(runSVMs(reg=val))
    for val in kernel:
        kernelF1Scores.append(runSVMs(kern=val))
    
    # plot reg metric results
    plt.plot(regularization, regF1Scores)
    plt.title("SVM: regularization vs F1 score")
    plt.xlabel("regularization")
    plt.ylabel("F1 score")
    plt.show()
    # plot kernel metric results
    plt.plot(kernel, kernelF1Scores)
    plt.title("SVM: kernel vs F1 score")
    plt.xlabel("kernel")
    plt.ylabel("F1 Score")
    plt.show()

    # train again with best parameters
    clf = svm.SVC(C=4, kernel='rbf')
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    
    # make confusion matrix
    f1Score = getMetrics(clf, y_test, predicted, clf.classes_, True, True)


def record(modId, num1k, centerFreq, plot):
    for i in range(3):
        num1k = int(num1k)
        plot = int(plot)
        # capture signal from rtl sdr
        sdr = RtlSdr()
        # 200 KHz for FM, 20 for AM
        fs = 1e6
        sdr.sample_rate = fs
        sdr.center_freq = float(centerFreq)*fs
        sdr.gain = 'auto'
        
        samples = sdr.read_samples(num1k*1024)
        sdr.close()
        
        string_fs = str(fs/1e6)+'MHz'
        fileName =  modId + '_' + centerFreq + '_' + string_fs + \
                    '_' + 'raw_data' + str(i) + '.npy'
        
        print("Writing Data")
        # write raw iq data to modify later if needed
        np.save(fileName, samples)
        
        '''
        # really no reason to save off spectrogram...
        saveSpec = False
        if saveSpec:
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
            plt.specgram(samples, Fs=1e6, cmap="rainbow")
            plt.title('Spectrogram')
            plt.xlabel("Time (s)")
            plt.ylabel("IF (MHz)")
            plt.show()


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("How to use this script:\n")
        print("Example: " +
              "python3.9 aml_project.py cnn fm_100.3_1.0MHz_raw_data.npy" +
              "fm_103.5_1.0MHz_raw_data.npy gsm_852.038_1.0MHz_raw_data.npy \n")
        print("--svm = train support vector machine model and test")
        print("    file names")
        print("--getdata = record raw iq data from SDR")
        print("    modulation - few character mod identifier")
        print("    num1k      - number of 1k samples to take")
        print("    centerFreq - center frequency in MHz")
        print("    plot       - show plots of just taken data")
        print("--plot = plots data from raw IQ files")
        print("    file names")
    else:
        path = sys.argv[2]
        fileNames = glob.glob(path + "/*.npy")
        if sys.argv[1] == 'svm':
            SVM(fileNames)
        elif sys.argv[1] == 'cnn':
            CNN(fileNames)
        elif sys.argv[1] == 'randomforest':
            randomForest(fileNames)
        elif sys.argv[1] == 'knn':
            KNN(fileNames)
        elif sys.argv[1] == 'getdata':
            record(*sys.argv[2:])
        elif sys.argv[1] == 'plot':
            fileNames = sys.argv[2:]
            for fileName in fileNames:
                plot(fileName)
        else:
            print("No argument matches your input")
