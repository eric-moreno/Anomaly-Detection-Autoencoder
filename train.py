# import libraries
import os
import pandas as pd
import numpy as np
import sys 
import setGPU
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
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

from model import autoencoder_LSTM, autoencoder_ConvLSTM, autoencoder_ConvDNN, autoencoder_DNN, autoencoder_Conv


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
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s'%outdir)

    # Load train and test data
    load = h5.File('data/default_simulated.hdf','r')
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:][:]

    #injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())][:]
    #y = injection_samples.reshape(-1)
     
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
        x = [filters(sample, freq) for sample in noise_samples]
        print('Done!')
        
    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(x)
    #X_test = scaler.transform(y.reshape(-1, 1))
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    joblib.dump(scaler, scaler_filename)
    
    # Data augmentation - only needed if not enough data
    '''
    x = []
    for sample in X_train: 
        if sample.shape[0]%timesteps != 0: 
            corrected_sample = sample[:-1*int(sample.shape[0]%timesteps)]
        #sliding_sample = np.array([sample[i:i+timesteps][:] for i in range(len(sample)-timesteps)])
        sliding_sample = np.array([corrected_sample[i:i+timesteps][:] for i in [int(timesteps/2)*n for n in range(int(len(corrected_sample)/(timesteps/2)) - 1)]])
        x.append(sliding_sample)
    X_train = np.array(x) 
    '''
    
    #Trim dataset to be batch-friendly and reshape into timestep format
    if X_train.shape[0]%timesteps != 0: 
        X_train = X_train[:-1*int(X_train.shape[0]%timesteps)]
    #if X_test.shape[0]%timesteps != 0: 
    #    X_test = X_test[:-1*int(X_test.shape[0]%timesteps)]
    
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(-1, timesteps, 1)
    #X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    #X_test = X_test.reshape(int(X_test.shape[0]/timesteps), timesteps, X_test.shape[1])
    #print("Test data shape:", X_test.shape)
 
    #Define model 
    model = autoencoder_DNN(X_train)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # fit the model to the data
    nb_epochs = 200
    batch_size = 1024
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('%s/best_model.hdf5'%(outdir), save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
    model.save('%s/last_model.hdf5'%(outdir))

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mae)')
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
    parser.add_argument("--timesteps", help="Number of timesteps passed to LSTM", action='store', dest='timesteps', default = 100)
    
    args = parser.parse_args()
    main(args)

