# import libraries
import os
import pandas as pd
import numpy as np
import sys 
import setGPU
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.utils import shuffle
import seaborn as sns
sns.set(color_codes=True)
import matplotlib.pyplot as plt
#%matplotlib inline
import h5py as h5
from gwpy.timeseries import TimeSeries

from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

from model_supervised import autoencoder_ConvDNN, autoencoder_DNN


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

    # Load train 
    load = h5.File('../../dataset/default_simulated.hdf', 'r')
    
    datapoints = 4000
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    train_data = np.concatenate((noise_samples, injection_samples))
    train_truth = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    train_data, train_truth = shuffle(train_data, train_truth)
    
    # Definining frequency in Hz instead of KHz
    if int(freq) == 2: 
        freq = 2048
    elif int(freq) == 4: 
        freq = 4096
    
    # With LIGO simulated data, the sample isn't pre-filtered so need to filter again. Real data
    # is not filtered yet. 
    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print('Sample Frequency: %s Hz'%(freq))
        x = [filters(sample, freq)[10240:12288] for sample in train_data]
        print('Done!')
    #7168:15360
    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(x)
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    joblib.dump(scaler, scaler_filename)
    

    print("Training data shape:", X_train.shape)

    #Define model 
    model = autoencoder_ConvDNN(X_train) 
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model to the data
    nb_epochs = 300
    batch_size = 16
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('%s/best_model.hdf5'%(outdir), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, train_truth, epochs=nb_epochs, batch_size=batch_size,
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

