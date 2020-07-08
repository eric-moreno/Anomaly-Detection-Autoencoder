# import libraries
import os
import requests
import pandas as pd
import numpy as np
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
from keras.models import load_model
from random import sample 
import argparse

from model import autoencoder_LSTM, autoencoder_Conv, autoencoder_DeepConv

def filters(array, sample_frequency):
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten()
    bp_data = white_data.bandpass(30, 400)
    return(bp_data.value)

def main(args):
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s'%outdir)
    
    load = h5.File('data/default.hdf','r')
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:1000]
    x = noise_samples.reshape(-1, 1)
    
    # Definining frequency in Hz instead of KHz
    freq = 1000*int(freq)
    
    if bool(int(filtered)): 
        x = filters(x, freq)
        
    # Load previous scaler and transform    
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    scaler = joblib.load(scaler_filename) 
    X_train = scaler.transform(x.reshape(-1, 1))
    
    # Trim dataset to be batch-friendly
    if X_train.shape[0]%timesteps != 0: 
        X_train = X_train[:-1*int(X_train.shape[0]%timesteps)]
        
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(int(X_train.shape[0]/timesteps), timesteps, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    
    # load the autoencoder network model
    model = load_model('%s/best_model.hdf5'%(outdir))
    
    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred = model.predict(X_train)
    print('Finished evaluating model on train data')
    X_pred = X_pred.reshape(X_pred.shape[0]*timesteps, X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred)

    scored = pd.DataFrame()
    Xtrain = X_train.reshape(X_train.shape[0]*timesteps, X_train.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
    plt.savefig('%s/loss_train_spread.jpg'%(outdir))
    
    threshold = scored.max()[0]
    print('The threshold is: %s'%(str(threshold)))
    
    # Evaluate on 10 test data events 
    injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())]
    times = load['injection_samples']['event_time']
    random_samples = sample(range(0, len(injection_samples)), 10)
    
    for random_sample in random_samples: 
        event = injection_samples[random_sample]
        time = times[random_sample] - 1000000000 # times start at 1000000000 for some reason
        
        if bool(int(filtered)): 
            event = filters(event, freq)
            
        X_test = scaler.transform(event.reshape(-1, 1))
        
        if X_test.shape[0]%timesteps != 0: 
            X_test = X_test[:-1*int(X_test.shape[0]%timesteps)]
            
        X_test = X_test.reshape(int(X_test.shape[0]/timesteps), timesteps, X_test.shape[1])
        X_pred_test = np.array(model.predict(X_test))
    
        X_pred_test = X_pred_test.reshape(X_pred_test.shape[0]*timesteps, X_pred_test.shape[2])
        X_pred_test = pd.DataFrame(X_pred_test)
        #X_pred_train.index = train.index
        Xtest = X_test.reshape(X_test.shape[0]*timesteps, X_test.shape[2])

        scored_test = pd.DataFrame()
        scored_test['Loss_mae'] = np.mean(np.abs(X_pred_test-Xtest), axis = 1)
        scored_test['Threshold'] = threshold
        scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
        #scored_test.plot(logy=True,  figsize=(16,9), ylim=[t/(1e2),threshold*(1e2)], color=['blue','red'])
        scored_test.plot(logy=False,  figsize=(16,9), color=['blue','red'])
        plt.axvline(5.5*2048, label='actual GW event', color='green') #Sampling rate of 2048 Hz with the event occuring 5.5 seconds into sample
        plt.legend(loc='upper left')
        plt.savefig('%s/test_threshold_%s_8sec.jpg'%(outdir, time))
        
    
if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()
    
    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("detector", help="Required output directory")
    parser.add_argument("--freq", help="Sampling frequency of detector in KHz", action='store', dest='freq', default = 4)
    parser.add_argument("--filtered", help="Apply LIGO's bandpass and whitening filters", action='store', dest='filtered', default = 1)
    parser.add_argument("--timesteps", help="Number of timesteps passed to LSTM", action='store', dest='timesteps', default = 100)
    
    args = parser.parse_args()
    main(args)