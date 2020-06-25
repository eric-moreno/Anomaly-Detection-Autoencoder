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
import argparse

from model import autoencoder_LSTM, autoencoder_DNN, autoencoder_Conv, autoencoder_DeepConv

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

    train_start = 1185939456
    num_train = 1
    train_files = [train_start + i*4096 for i in range(num_train)]
    x = np.array([])
    
    for file in train_files: 
        load = h5.File('/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.%sk/hdf.v1/%s/%s/%s-%s_GWOSC_O2_%sKHZ_R1-%s-4096.hdf5'%(str(freq),detector, str(train_start), detector[0], detector, str(freq), str(file)))
        x = np.concatenate((x, load['strain']['Strain'][()]), axis=0)
    '''    
    test_start = 1186988032
    num_test = 1
    test_files = [test_start + 5*4096 + i*4096 for i in range(num_test)]
    y = np.array([])

    for file in test_files: 
        load = h5.File('/cvmfs/gwosc.osgstorage.org/gwdata/O2/strain.%sk/hdf.v1/%s/%s/%s-%s_GWOSC_O2_%sKHZ_R1-%s-4096.hdf5'%(str(freq),detector, str(test_start), detector[0], detector, str(freq), str(file)))
        y = np.concatenate((y, load['strain']['Strain'][()]), axis=0)
    '''
    # Definining frequency in Hz instead of KHz
    freq = 1000*int(freq)
    
    if bool(filtered): 
        x = filters(x, freq)
        #y = filters(y, freq)
        
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    scaler = joblib.load(scaler_filename) 
    X_train = scaler.fit_transform(x.reshape(-1, 1))
    #X_test = scaler.transform(y.reshape(-1, 1))
    
    #timesteps = int(args.timesteps)
    #X_test = X_test[int((370.4-30)*freq):int((370.4+30)*freq)]
    
    if X_train.shape[0]%timesteps != 0: 
        X_train = X_train[:-1*int(X_train.shape[0]%timesteps)]
    
   # if X_test.shape[0]%timesteps != 0: 
   #     X_test = X_test[:-1*int(X_test.shape[0]%timesteps)]
    # reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(int(X_train.shape[0]/timesteps), timesteps, X_train.shape[1])
    #X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    print("Training data shape:", X_train.shape)
    #X_test = X_test.reshape(int(X_test.shape[0]/timesteps), timesteps, X_test.shape[1])
    #X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    #print("Test data shape:", X_test.shape)

    # load the autoencoder network model
    model = load_model('%s/best_model.hdf5'%(outdir))
    #model.load('%s/best_model.hdf5'%(outdir))
    
    '''
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred = model.predict(X_train)
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
    '''
    threshold = 0.59
    print('The threshold is: %s'%(str(threshold)))
    
    
    GW_GPS_confirmed = [1167559936.6, 1180922494.5, 1185389807.3, 1186302519.8, 1186741861.5, 1187008882.4, 1187058327.1, 1187529256.5]
    
    for t0 in GW_GPS_confirmed: 
        from gwosc.locate import get_urls
        try:
            url = get_urls(detector, t0, t0)[-1]
        except: 
            continue 
        print('Downloading: ' , url)
        fn = os.path.basename(url)
        with open(fn,'wb') as strainfile:                 
            straindata = requests.get(url)
            strainfile.write(straindata.content)
            
        strain = TimeSeries.read(fn,format='hdf5.losc')
        center = int(t0)
        strain = strain.crop(center-30, center+30)
        
        white_data = strain.whiten()
        bp_data = white_data.bandpass(30, 400)
        X_test = scaler.transform(bp_data.value.reshape(-1, 1))
        
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
        plt.savefig('%s/test_threshold_%s_pm30sec.jpg'%(outdir, t0))
    
    '''
    print('Evaluating Model on test data. This make take a while...')
    ### Evaluating on test data to find threshold ### 
    X_pred_test = np.array(model.predict(X_test))
    X_pred_test = X_pred_test.reshape(X_pred_test.shape[0]*timesteps, X_pred_test.shape[2])
    X_pred_test = pd.DataFrame(X_pred_test)
    #X_pred_train.index = train.index
    Xtest = X_test.reshape(X_test.shape[0]*timesteps, X_test.shape[2])
    
    scored_test = pd.DataFrame()
    scored_test['Loss_mae'] = np.mean(np.abs(X_pred_test-Xtest), axis = 1)
    scored_test['Threshold'] = threshold
    scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
    scored_test.plot(logy=True,  figsize=(16,9), ylim=[threshold/(1e2),threshold*(1e2)], color=['blue','red'])
    plt.savefig('%s/test_threshold_pm30sec.jpg'%(outdir))
    '''
    
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