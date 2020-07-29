# import libraries
import os
import sys
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
from keras.models import load_model
import argparse
from sklearn.metrics import roc_curve, auc, accuracy_score

from model import autoencoder_LSTM, autoencoder_Conv, autoencoder_DeepConv

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
    
    load = h5.File('../../dataset/default_simulated.hdf', 'r')
    
    if int(freq) == 2: 
        freq = 2048
    elif int(freq) == 4: 
        freq = 4096
        
    if freq%2048 != 0: 
        print('WARNING: not a supported sampling frequency for simulated data')
        print('Sampling Frequency: %s'%(freq))
    
    n_test_events = 1000
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:][-n_test_events:]
    injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())][:][-n_test_events:]
    test_data = np.concatenate((noise_samples, injection_samples))
    test_truth = np.concatenate((np.zeros(n_test_events), np.ones(n_test_events)))
    
    if bool(int(filtered)):
        print('filtering data with whitening and bandpass')
        test_data = [filters(sample, freq)[10240:12288] for sample in test_data]
        print('Done!')
        
    # Load previous scaler and transform    
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    scaler = joblib.load(scaler_filename) 
    X_test = scaler.transform(test_data)
    
    print("Testing data shape:", X_test.shape)
    
    model = load_model('%s/best_model.hdf5'%(outdir))
    X_pred_test = model.predict(X_test)
    
    directory_list = [outdir]
    names = ['CNN+DNN']
    
    plt.figure()
    for name, directory in zip(names, directory_list): 
        print('Determining performance for: %s'%(name))
        fpr, tpr, thresholds = roc_curve(test_truth, X_pred_test)
        plt.plot(fpr, tpr, lw=2, label='%s (auc = %0.2f)'%(name, auc(fpr, tpr)))
        print('Done!')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([1e-4, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('LIGO Supervised GW-Detection')
    plt.legend(loc="lower right")
    plt.savefig('%s/ROC_curve_log.jpg'%(outdir))
    
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
