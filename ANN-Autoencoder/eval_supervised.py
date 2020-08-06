# import libraries
import os
import sys
import requests
import pandas as pd
import numpy as np
import setGPU
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
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
    
    load = h5.File('../../dataset/240k_1sec_L1_GWdistributed.h5', 'r')

    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

    datapoints = 120000
    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    targets = np.transpose(np.array([gw, noise]))

    X = load['data'][:]
    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(X, targets, test_size=0.2, random_state=42)
    class_names = np.array(['noise', 'GW'], dtype=str)

    print(train_data.shape)
    # Reshape inputs
    train_data = train_data.reshape((train_data.shape[0], 1, -1))
    print("Train data shape:", train_data.shape)
    #train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
    print("Train labels data shape:", train_truth.shape)
    test_data = test_data.reshape((test_data.shape[0], 1, -1))
    print("Test data shape:", test_data.shape)
    #test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
    print("Test labels data shape:", test_truth.shape)
    
    # Evaluate model
    model = load_model('%s/best_model.hdf5'%(outdir))
    X_pred_test = model.predict(test_data)
    
    directory_list = [outdir]
    names = ['CNN Nengo Architecture']
    predictions = [X_pred_test]
    
    # ROC Curve Plot
    plt.figure()
    for name, directory, pred in zip(names, directory_list, predictions): 
        print('Determining performance for: %s'%(name))
        fpr, tpr, thresholds = roc_curve(np.argmax(test_truth, axis=-1), np.argmax(pred, axis=-1))
        plt.plot(fpr, tpr, lw=2, label='%s (auc = %0.2f)'%(name, auc(fpr, tpr)))
        print('Accuracy: %s'%(accuracy_score(np.argmax(test_truth, axis=-1), np.argmax(pred, axis=-1))))
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
              
    # SNR vs Efficiency plot
    plt.figure()
    bins = 30
    SNR_max = max(SNR)
    SNR_min = min(SNR)
    SNR_bins = [[] for i in range(int(bins))]
    for name, directory, pred in zip(names, directory_list, predictions): 
        for i in range(int(len(pred)/2)): 
            if abs(pred[i] - test_truth[i]) <= 0.5:
                SNR_bins[int((SNR[i] - SNR_min)/((SNR_max-SNR_min)/bins)) - 1].append(1)
            else: 
                SNR_bins[int((SNR[i] - SNR_min)/((SNR_max-SNR_min)/bins)) - 1].append(0)
                
    
            
    x = [sum(i)/len(i) for i in SNR_bins[:-10]]
    plt.plot(x)
    plt.xlabel('SNR')
    plt.ylabel('True Positive Rate')
    plt.savefig('%s/SNR_efficiency.jpg'%(outdir))
    
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
