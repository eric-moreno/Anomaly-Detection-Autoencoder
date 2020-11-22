""" Evaluation of model trained for anomaly detection. """

import os, sys
import argparse
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
import tensorflow as tf
from gwpy.timeseries import TimeSeries
from keras.models import load_model
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError
from random import sample as RandSample
from sklearn.metrics import auc
import setGPU 

sns.set(color_codes=True)

def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value

def TPR_FPR_arrays(noise_array, injection_array, model_outdir, steps, num_entries=400): 
    # load the autoencoder network model
    model = load_model('%s/best_model.hdf5'%(model_outdir))
    '''
    x = []
    for event in range(len(noise_array)): 
        if noise_array[event].shape[0]%steps != 0: 
            #x.append(noise_array[event][:-1*int(noise_array[event].shape[0]%steps)])
            x.append(noise_array[event][:-92])
    noise_array = np.array(x).reshape(-1, steps, 1)
    
    x = []
    for event in range(len(injection_array)): 
        if injection_array[event].shape[0]%steps != 0: 
            #x.append(injection_array[event][:-1*int(injection_array[event].shape[0]%steps)])
            x.append(injection_array[event][:-92])
    injection_array = np.array(x).reshape(-1, steps, 1)
    '''
     
    noise_array = noise_array.reshape(-1, steps, 1)
    injection_array = injection_array.reshape(-1, steps, 1)
    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred_noise = model.predict(noise_array)
    print('Finished evaluating model on train data')
    
    n_noise_events = 10000
    # Determine thresholds for FPR quantiles
    loss_fn = MeanSquaredError(reduction='none')
    losses = loss_fn(noise_array, X_pred_noise).numpy()
    averaged_losses = np.mean(losses, axis=1).reshape(n_noise_events, -1)
    max_losses = [np.max(event) for event in averaged_losses]

    roc_steps = num_entries
    FPRs = np.logspace(-3, 0, roc_steps)
    thresholds = [np.quantile(max_losses, 1.0-fpr) for fpr in FPRs]
    
    print('Evaluating Model on test data. This make take a while...')
    X_pred_injection = model.predict(injection_array)
    print('Finished evaluating model on test data')
    
    n_injection_events = 10000
    losses = loss_fn(injection_array, X_pred_injection).numpy()
    averaged_losses = np.mean(losses, axis=1).reshape(n_injection_events, -1)
    
    # For each event determine whether GW was detected at a given FPR threshold
    gw_pred = [[] for i in range(roc_steps)]
    for i in range(len(averaged_losses)):
        batch_loss = averaged_losses[i]

        for fpr in range(len(FPRs)):
            if np.max(batch_loss) > thresholds[fpr]: 
                gw_pred[fpr].append(1)
            else: 
                gw_pred[fpr].append(0)

    # Calculate corresponding TPR
    TPRs = [float(np.sum(gw_pred[fpr]))/n_injection_events for fpr in range(len(FPRs))]
    return(TPRs, FPRs)

def TPR_FPR_arrays_doubledetector(noise_array_L1, injection_array_L1, noise_array_H1, injection_array_H1, model_outdir, steps, num_entries=400): 
    # load the autoencoder network model
    model = load_model('%s/best_model.hdf5'%(model_outdir))
     
    noise_array_L1 = noise_array_L1.reshape(-1, steps, 1)
    injection_array_L1 = injection_array_L1.reshape(-1, steps, 1)
    noise_array_H1 = noise_array_H1.reshape(-1, steps, 1)
    injection_array_H1 = injection_array_H1.reshape(-1, steps, 1)
    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred_noise_L1 = model.predict(noise_array_L1)
    X_pred_noise_H1 = model.predict(noise_array_H1)
    print('Finished evaluating model on train data')
    
    n_noise_events = 10000
    # Determine thresholds for FPR quantiles
    loss_fn = MeanSquaredError(reduction='none')
    losses_L1 = loss_fn(noise_array_L1, X_pred_noise_L1).numpy()
    losses_H1 = loss_fn(noise_array_H1, X_pred_noise_H1).numpy()
    averaged_losses_L1 = np.mean(losses_L1, axis=1).reshape(n_noise_events, -1)
    averaged_losses_H1 = np.mean(losses_H1, axis=1).reshape(n_noise_events, -1)
    max_losses = [np.max(event_L1) + np.max(event_H1) for event_L1, event_H1 in zip(averaged_losses_L1, averaged_losses_H1)]
    
    roc_steps = num_entries
    FPRs = np.logspace(-3, 0, roc_steps)
    thresholds = [np.quantile(max_losses, 1.0-fpr) for fpr in FPRs]
    
    print('Evaluating Model on test data. This make take a while...')
    X_pred_injection_L1 = model.predict(injection_array_L1)
    X_pred_injection_H1 = model.predict(injection_array_H1)
    print('Finished evaluating model on test data')
    
    n_injection_events = 10000
    losses_L1 = loss_fn(injection_array_L1, X_pred_injection_L1).numpy()
    losses_H1 = loss_fn(injection_array_H1, X_pred_injection_H1).numpy()
    averaged_losses_L1 = np.mean(losses_L1, axis=1).reshape(n_injection_events, -1)
    averaged_losses_H1 = np.mean(losses_H1, axis=1).reshape(n_injection_events, -1)
    
    # For each event determine whether GW was detected at a given FPR threshold
    gw_pred = [[] for i in range(roc_steps)]
    for i in range(len(averaged_losses_L1)):
        batch_loss_L1 = averaged_losses_L1[i]
        batch_loss_H1 = averaged_losses_H1[i]
        for fpr in range(len(FPRs)):
            if np.max(batch_loss_H1)+np.max(batch_loss_L1) > thresholds[fpr]: 
                gw_pred[fpr].append(1)
            else: 
                gw_pred[fpr].append(0)

    # Calculate corresponding TPR
    TPRs = [float(np.sum(gw_pred[fpr]))/n_injection_events for fpr in range(len(FPRs))]
    return(TPRs, FPRs)

def main(args):
    """ Main function to evaluate the model """
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s' % outdir)

    load_L1 = h5.File('../../dataset/default_BIGsim_testtrain_L1.h5', 'r')
    load_H1 = h5.File('../../dataset/default_BIGsim_testtrain_H1.h5', 'r')
    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')
    '''
    n_noise_events = 5000
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:][-n_noise_events:]

    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print(f'Sample Frequency: {freq} Hz')
        x_noise = [filters(sample, freq) for sample in noise_samples]
        print('Filtering completed')

    # Load previous scaler and transform
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    scaler = joblib.load(scaler_filename)
    X_train = scaler.transform(x_noise)

    print("Training data shape:", X_train.shape)
    
    n_injection_events = 5000
    injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())][:][:n_injection_events]

    if bool(int(filtered)):
        print('filtering data with whitening and bandpass')
        x_injection = [filters(sample, freq) for sample in injection_samples]
        print('Done!')
        
    # Normalize the data
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    scaler = joblib.load(scaler_filename) 
    X_test = scaler.transform(x_injection)

    print("Testing data shape:", X_test.shape)
    '''
    X_test_L1 = load_L1['injection'][:10000, :16092]
    X_train_L1 = load_L1['noise'][300000:310000, :16092]
    X_test_H1 = load_H1['injection'][:10000, :16092]
    X_train_H1 = load_H1['noise'][300000:310000, :16092]
    
    #directory_list = [outdir]
    #names = ['LSTM Autoencoder']
    #timesteps = [timesteps]
    directory_list = ['BIGsimdata_L1_2KHz_unsupervised_filtered_DNN', 'BIGsimdata_L1_2KHz_unsupervised_filtered_ConvDNN']#, 'BIGsimdata_L1_2KHz_unsupervised_filtered_LSTM']
    names = ['DNN Autoencoder', 'CNN-DNN Autoencoder']#, 'LSTM Autoencoder']
    timesteps = [100, 108, 100]
    FPR_set = []
    TPR_set = []
    
    for name, directory, timestep in zip(names, directory_list, timesteps): 
        print('Determining performance for: %s'%(name))
        if timestep == 100: 
            #TPR, FPR = TPR_FPR_arrays_doubledetector(X_train_L1[:, :16000], X_test_L1[:, :16000], X_train_H1[:, :16000], X_test_H1[:, :16000], directory, timestep)
            TPR, FPR = TPR_FPR_arrays(X_train_H1[:, :16000], X_test_H1[:, :16000], directory, timestep)
        else: 
            #TPR, FPR = TPR_FPR_arrays_doubledetector(X_train_L1, X_test_L1, X_train_H1, X_test_H1, directory, timestep)
            TPR, FPR = TPR_FPR_arrays(X_train_H1, X_test_H1, directory, timestep)
        TPR_set.append(TPR)
        FPR_set.append(FPR)
        print('Done!')
    
    plt.figure()
    lw = 2
    for FPRs, TPRs, name in zip(FPR_set, TPR_set, names):
        plt.plot(FPRs, TPRs,
             lw=lw, label='%s (auc = %0.5f)'%(name, auc(FPRs, TPRs)))
    plt.plot([0, 1], [0, 1], lw=lw, linestyle='--')
    plt.xlim([1e-3, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('LIGO Unsupervised Autoencoder Anomaly Detection')
    plt.legend(loc="upper left")
    plt.savefig('%s/ROC_curve_log_1.jpg'%(outdir))

    sys.exit()
    ### Enable if needed - these are additional plots to check if methods are working in unsupervised learning approach###
    
    
    random_samples = RandSample(range(0, len(X_test)), 10)
    for random_sample in random_samples: 
        event = X_test[random_sample, :16000]
        event = event.reshape(-1, int(args.timesteps), 1)
        time = random_sample
        loss_fn = MeanSquaredError(reduction='none')

        model = load_model('%s/best_model.hdf5'%(outdir))
        X_pred = model.predict(event)
        
        losses = loss_fn(event, X_pred).eval(session=tf.compat.v1.Session())
        print(losses)
        print(losses.shape)
        batch_loss = np.mean(losses, axis=1)
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(batch_loss)
        plt.axvline(112.64, label='actual GW event', color='green')
        plt.xlabel('Timestep')
        plt.ylabel('Loss')
        plt.title('LSTM Autoencoder')
        #plt.axhline(threshold, label='GW event threshold', color='red')
        plt.legend(loc='upper left')
        plt.savefig('%s/batchloss_%s.jpg'%(outdir,time))
        
        '''
        X_pred_test = np.array(model.predict(event))
        
        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(event.reshape(-1)[int(2048*5.5) - 300:int(2048*5.5) + 300], label='truth')
        ax.plot(X_pred_test.reshape(-1)[int(2048*5.5)- 300:int(2048*5.5) + 300], label='predict')
        plt.legend(loc='upper left')
        plt.title('LSTM Autoencoder')
        plt.savefig('%s/middle30ms_%s.jpg'%(outdir,time))
        
        print(X_pred_test.shape)
        X_pred_test = X_pred_test.reshape(X_pred_test.shape[0]*timesteps, X_pred_test.shape[2])
        
        #X_pred_train.index = train.index
        Xtest = event.reshape(event.shape[0]*timesteps, event.shape[2])

        X_pred_test = pd.DataFrame(X_pred_test)
        scored_test = pd.DataFrame()
        scored_test['Loss_mae'] = np.mean(np.abs(X_pred_test-Xtest), axis = 1)
        #scored_test['Threshold'] = threshold
        #scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
        #scored_test.plot(logy=True,  figsize=(16,9), ylim=[t/(1e2),threshold*(1e2)], color=['blue','red'])
        scored_test.plot(logy=False,  figsize=(16,9), color=['blue','red'])
        plt.axvline(5.5*2048, label='actual GW event', color='green') #Sampling rate of 2048 Hz with the event occuring 5.5 seconds into sample
        plt.legend(loc='upper left')
        plt.savefig('%s/test_threshold_%s_8sec.jpg'%(outdir, time))
        '''

    sys.exit()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("detector", help="LIGO Detector")

    # Additional arguments
    parser.add_argument("--freq", help="Sampling frequency of detector in KHz",
                        action='store', dest='freq', default=2)
    parser.add_argument("--filtered", help="Apply LIGO's bandpass and whitening filters",
                        action='store', dest='filtered', default=1)
    parser.add_argument("--timesteps", help="Number of timesteps passed to LSTM",
                        action='store', dest='timesteps', default=100)

    args = parser.parse_args()
    main(args)
