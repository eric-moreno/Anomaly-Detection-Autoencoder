# import libraries
import os
import pandas as pd
import numpy as np
import sys 
import setGPU
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline
import h5py as h5
from gwpy.timeseries import TimeSeries

from numpy.random import seed
import tensorflow as tf

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

from model_supervised import autoencoder_ConvDNN, autoencoder_DNN, autoencoder_ConvDNN_Nengo, model_LSTM, model_ConvDNN, autoencoder_ConvDNNLSTM, model_LSTM_seq, model_ConvLSTM_1, model_ConvLSTM_2


def filters(array, sample_frequency):
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4,fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return(bp_data.value)
    
def main(args):
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    os.system('mkdir -p %s'%outdir)

    load = h5.File(f'../../dataset/default_BBH_{detector}.h5', 'r')
    
    load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_12seed.hdf', 'r')
    X_train_L1 = load['noise_samples']['l1_strain'][:]
    X_train_H1 = load['noise_samples']['h1_strain'][:]
    X_train = np.concatenate((X_train_L1, X_train_H1))

    load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_13seed.hdf', 'r')
    X_train_L1 = load['noise_samples']['l1_strain'][:]
    X_train_H1 = load['noise_samples']['h1_strain'][:]
    X_train = np.concatenate((X_train, X_train_L1))
    X_train = np.concatenate((X_train, X_train_H1))
    
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    
    X_train[:, index:index+int(2.5*2048) for index in np.random.randint(int(3.5*2048),5*2048, size=len(noise_samples))]
    
    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

    datapoints = 75000
    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    targets = np.transpose(np.array([gw, noise]))

    X = np.concatenate((load['injection'][:datapoints], load['noise'][:datapoints]))
    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(X, targets, test_size=0.2, random_state=42)
    class_names = np.array(['noise', 'GW'], dtype=str)

    print(train_data.shape)
    # Reshape inputs
    #train_data = train_data.reshape((train_data.shape[0], 1, -1))
    train_data = train_data.reshape((train_data.shape[0], -1))
    print("Train data shape:", train_data.shape)
    #train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
    print("Train labels data shape:", train_truth.shape)
    #test_data = test_data.reshape((test_data.shape[0], 1, -1))
    test_data = test_data.reshape((test_data.shape[0], -1))
    print("Test data shape:", test_data.shape)
    #test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
    print("Test labels data shape:", test_truth.shape)
    
    #Define model 
    model = model_LSTM(train_data) 
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()

    # fit the model to the data
    nb_epochs = 300
    batch_size = 1
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('%s/best_model_%s.hdf5'%(outdir, detector), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(train_data, train_truth, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
    model.save('%s/last_model.hdf5'%(outdir))

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig('%s/loss.jpg'%(outdir))

if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("detector", help="LIGO Detector")
    parser.add_argument("--freq", help="Sampling frequency of detector in KHz", action='store', dest='freq', default = 2)
    parser.add_argument("--filtered", help="Apply LIGO's bandpass and whitening filters", action='store', dest='filtered', default = 1)
    
    args = parser.parse_args()
    main(args)

