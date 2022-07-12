""" Evaluation of model trained for anomaly detection. """

import os, sys, psutil
import argparse
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import h5py as h5
import tensorflow as tf
from gwpy.timeseries import TimeSeries
from keras.models import load_model
from tensorflow.keras.losses import mean_absolute_error, MeanAbsoluteError, mean_squared_error, MeanSquaredError
from sklearn.model_selection import train_test_split
from random import sample as RandSample
from keras.models import Model
from sklearn.metrics import auc
import setGPU 
from scipy.spatial import distance
from sklearn.metrics import roc_curve, auc, accuracy_score

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rcParams['font.size'] = 18
rcParams['text.latex.preamble'] = [
#       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
#       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
rc('text', usetex=True)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value

def augmentation(X_train, timesteps):
    """ Data augmentation process used to extend dataset """
    x = []
    for sample in X_train:
        if sample.shape[0] % timesteps != 0:
            sample = sample[:-1 * int(sample.shape[0] % timesteps)]
        
        divide_into = 10
        
        sliding_sample = np.array([sample[i:i + timesteps] for i in [int(timesteps / divide_into) * n for n in range(
                                    int(len(sample) / (timesteps / divide_into)) - (divide_into - 1))]])
        x.append(sliding_sample)
    return np.array(x)

def threshold_given_FPR(noise_array, injection_array, model_outdir, steps, FPR): 
    # load the autoencoder network model
    model = load_model('%s/best_model_%s.hdf5'%(model_outdir, args.detector))
    
    n_noise_events = np.shape(noise_array)[0]
    n_injection_events = np.shape(injection_array)[0]
    
    noise_array = noise_array.reshape(-1, steps, 1)
    injection_array = injection_array.reshape(-1, steps, 1)

    print(np.shape(noise_array))
    prin(np.shape(injection_array))
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred_noise = model.predict(noise_array)
    print('Finished evaluating model on train data')
    
    #n_noise_events = 100000
    # Determine thresholds for FPR quantiles
    loss_fn = MeanSquaredError(reduction='none')
    losses = loss_fn(noise_array, X_pred_noise).numpy()
    averaged_losses = np.mean(losses, axis=1).reshape(n_noise_events, -1)
    max_losses = [np.max(event) for event in averaged_losses]
    threshold = np.quantile(max_losses, 1.0-FPR) 
    return(threshold)


def TPR_FPR_arrays(noise_array, injection_array, model_outdir, steps, num_entries=100): 
    # load the autoencoder network model
    #model = load_model('%s/best_model_%s.hdf5'%(model_outdir, args.detector))
    model = load_model('%s/best_model.hdf5'%(model_outdir))
    
    print(noise_array.shape)
    noise_array = augmentation(noise_array, steps)
    injection_array = augmentation(injection_array, steps)
    print(noise_array.shape)
    
    n_noise_events = np.shape(noise_array)[0]
    n_injection_events = np.shape(injection_array)[0]
    
    noise_array = noise_array.reshape(-1, steps, 1)
    injection_array = injection_array.reshape(-1, steps, 1)

    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    X_pred_noise = model.predict(noise_array)
    print('Finished evaluating model on train data')
    
    #n_noise_events = 100000
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
    
    #n_injection_events = 10000
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
    index_FPR_1percent = 0 
    for i in FPRs: 
        if i < 0.01: 
            index_FPR_1percent+=1
    print('TPR for classifier at FPR=0.01: %s'%(TPRs[index_FPR_1percent]))
    
    index_FPR_10percent = 0 
    for i in FPRs: 
        if i < 0.1: 
            index_FPR_10percent+=1
    print('TPR for classifier at FPR=0.1: %s'%(TPRs[index_FPR_10percent]))
    
    return(TPRs, FPRs)


def TPR_FPR_arrays_doubledetector(noise_array_L1, injection_array_L1, noise_array_H1, injection_array_H1, model_outdir, steps, num_entries=100): 
    # load the autoencoder network model
    
    if clip: 
        model_L1 = load_model('%s/best_clip_standard_l1.hdf5'%(model_outdir))
        model_H1 = load_model('%s/best_clip_standard_l1.hdf5'%(model_outdir))
    if badremoved:     
        model_L1 = load_model('%s/best_badremoved_standard_l1.hdf5'%(model_outdir))
        model_H1 = load_model('%s/best_badremoved_standard_l1.hdf5'%(model_outdir))
    
    n_noise_events = np.shape(noise_array_L1)[0]
    n_injection_events = np.shape(injection_array_L1)[0]
    
    noise_array_L1 = noise_array_L1.reshape(-1, steps, 1)
    injection_array_L1 = injection_array_L1.reshape(-1, steps, 1)
    noise_array_H1 = noise_array_H1.reshape(-1, steps, 1)
    injection_array_H1 = injection_array_H1.reshape(-1, steps, 1)
    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
    print(noise_array_L1)
    X_pred_noise_L1 = model_L1.predict(noise_array_L1)
    X_pred_noise_H1 = model_H1.predict(noise_array_H1)
    
    #X_pred_noise_L1 = np.load('%s/X_pred_noise_BNS_L1_%s.npy'%(model_outdir, model_outdir))
    #X_pred_noise_H1 = np.load('%s/X_pred_noise_BNS_H1_%s.npy'%(model_outdir, model_outdir))
    
    print('Finished evaluating model on train data')
    
    # Determine thresholds for FPR quantiles
    loss_fn = MeanSquaredError(reduction='none')
    losses_L1 = loss_fn(noise_array_L1, X_pred_noise_L1).numpy()
    losses_H1 = loss_fn(noise_array_H1, X_pred_noise_H1).numpy()
    
    del X_pred_noise_L1,X_pred_noise_H1, noise_array_L1, noise_array_H1
    averaged_losses_L1 = np.mean(losses_L1, axis=1).reshape(n_noise_events, -1)
    averaged_losses_H1 = np.mean(losses_H1, axis=1).reshape(n_noise_events, -1)
    
    plt.figure()
    
    max_loss_L1 = [np.max(event_L1) for event_L1 in averaged_losses_L1]
    max_loss_H1 = [np.max(event_H1) for event_H1 in averaged_losses_H1]
    
    plt.scatter(max_loss_L1, max_loss_H1, alpha = 0.5, color = 'blue', label = 'noise')
    
    max_losses = [np.max(event_L1) + np.max(event_H1) for event_L1, event_H1 in zip(averaged_losses_L1, averaged_losses_H1)]
    
    roc_steps = num_entries
    FPRs = np.logspace(-np.log(20000)/np.log(10), 0, roc_steps)
    #FPRs = np.logspace(-3, 0, roc_steps)
    FPRs_table = [0.01, 0.0001]
    thresholds = [np.quantile(max_losses, 1.0-fpr) for fpr in FPRs]
    thresholds_table = [np.quantile(max_losses, 1.0-fpr) for fpr in FPRs_table]
    thresholds_L1 = [np.quantile(max_loss_L1, 1.0-fpr) for fpr in FPRs]
    thresholds_H1 = [np.quantile(max_loss_H1, 1.0-fpr) for fpr in FPRs]
    
    print('Evaluating Model on test data. This make take a while...')
    X_pred_injection_L1 = model_L1.predict(injection_array_L1)
    X_pred_injection_H1 = model_H1.predict(injection_array_H1)
    
    #X_pred_injection_L1 = np.load('%s/X_pred_injection_BNS_L1_%s.npy'%(model_outdir, model_outdir))    
    #X_pred_injection_H1 = np.load('%s/X_pred_injection_BNS_H1_%s.npy'%(model_outdir, model_outdir))
    
    print('Finished evaluating model on test data')

    losses_L1 = loss_fn(injection_array_L1, X_pred_injection_L1).numpy()
    losses_H1 = loss_fn(injection_array_H1, X_pred_injection_H1).numpy()
    
    del X_pred_injection_L1,X_pred_injection_H1, injection_array_L1, injection_array_H1
    
    averaged_losses_L1 = np.mean(losses_L1, axis=1).reshape(n_injection_events, -1)
    averaged_losses_H1 = np.mean(losses_H1, axis=1).reshape(n_injection_events, -1)
    
    max_loss_L1 = [np.max(event_L1) for event_L1 in averaged_losses_L1]
    max_loss_H1 = [np.max(event_H1) for event_H1 in averaged_losses_H1]
    
    plt.scatter(max_loss_L1, max_loss_H1, alpha = 0.1, color = 'red', label = 'injection')
    plt.xlabel('L1 Loss')
    plt.ylabel('H1 Loss')
    plt.title('LSTM AE simulated waveforms')
    plt.legend()
    plt.savefig(outdir+'/2dspace.jpg')
    
    # For each event determine whether GW was detected at a given FPR threshold
    gw_pred = [[] for i in range(roc_steps)]
    gw_pred_L1 = []
    gw_pred_H1 = []
    gw_pred_table = [[] for i in range(2)]
    for i in range(len(averaged_losses_L1)):
        batch_loss_L1 = averaged_losses_L1[i]
        batch_loss_H1 = averaged_losses_H1[i]
        for fpr in range(len(FPRs)):
            if np.max(batch_loss_H1)+np.max(batch_loss_L1) > thresholds[fpr]: 
                gw_pred[fpr].append(1)
            else: 
                gw_pred[fpr].append(0)
        
            #if np.max(batch_loss_H1) > thresholds_H1[fpr] and np.max(batch_loss_L1) > thresholds_L1[fpr]:
            #    gw_pred[fpr].append(1)
            #else: 
            #    gw_pred[fpr].append(0)
    
        for fpr in range(len(FPRs_table)):
            if np.max(batch_loss_H1)+np.max(batch_loss_L1) > thresholds_table[fpr]: 
                gw_pred_table[fpr].append(1)
            else: 
                gw_pred_table[fpr].append(0)
        
        if np.max(batch_loss_L1) > thresholds_L1[0]: 
            gw_pred_L1.append(1)
        else: 
            gw_pred_L1.append(0)
        
        if np.max(batch_loss_H1) > thresholds_H1[0]: 
            gw_pred_H1.append(1)
        else: 
            gw_pred_H1.append(0)
    
    print(np.corrcoef([gw_pred_H1, gw_pred_L1]))
          
    print('TPR for a 0.01 FPR is: %s'%(float(np.sum(gw_pred_table[0]))/n_injection_events))
    print('TPR for a 0.0001 FPR is: %s'%(float(np.sum(gw_pred_table[1]))/n_injection_events))
    
    # Calculate corresponding TPR
    TPRs = [float(np.sum(gw_pred[fpr]))/n_injection_events for fpr in range(len(FPRs))]
    return(TPRs, FPRs)

def TPR_FPR_arrays_latentspace(noise_array_L1, injection_array_L1, noise_array_H1, injection_array_H1, model_outdir, steps, num_entries=100): 
    # load the autoencoder network model
    
    if clip: 
        model = load_model('%s/best_clip_standard_l1h1.hdf5'%(model_outdir))
    if badremoved:     
        model_l1 = load_model('%s/best_badremoved_standard_l1h1.hdf5'%(model_outdir))
        model_h1 = load_model('%s/best_badremoved_standard_l1h1.hdf5'%(model_outdir))
    
    n_noise_events = np.shape(noise_array_L1)[0]
    n_injection_events = np.shape(injection_array_L1)[0]
    
    noise_array_L1 = noise_array_L1.reshape(-1, steps, 1)
    injection_array_L1 = injection_array_L1.reshape(-1, steps, 1)
    noise_array_H1 = noise_array_H1.reshape(-1, steps, 1)
    injection_array_H1 = injection_array_H1.reshape(-1, steps, 1)
    
    intermediate_layer_model_l1 = Model(inputs=model_l1.input,
                                 outputs=model_l1.layers[2].output)
    
    intermediate_layer_model_h1 = Model(inputs=model_h1.input,
                                 outputs=model_h1.layers[2].output)
    
    intermediate_output_L1_noise = intermediate_layer_model_l1.predict(noise_array_L1, verbose=1).reshape((n_noise_events, -1))
    intermediate_output_H1_noise = intermediate_layer_model_h1.predict(noise_array_H1, verbose=1).reshape((n_noise_events, -1))
    
    intermediate_output_L1_injection = intermediate_layer_model_l1.predict(injection_array_L1, verbose=1).reshape((n_injection_events, -1))
    intermediate_output_H1_injection = intermediate_layer_model_h1.predict(injection_array_H1, verbose=1).reshape((n_injection_events, -1))

    JSD_time_noise = [distance.euclidean(intermediate_output_L1_noise[i], intermediate_output_H1_noise[i]) 
                      for i in range(len(intermediate_output_L1_noise))]
    JSD_time_injection = [distance.euclidean(intermediate_output_L1_injection[i], intermediate_output_H1_injection[i]) 
                          for i in range(len(intermediate_output_L1_injection))]
    
    max_JSD = [np.max(event) for event in JSD_time_noise]
    roc_steps = num_entries
    FPRs = np.logspace(-3, 0, roc_steps)
    thresholds = [np.quantile(max_JSD, 1.0-fpr) for fpr in FPRs]
        # For each event determine whether GW was detected at a given FPR threshold
    gw_pred = [[] for i in range(roc_steps)]
    for i in range(len(JSD_time_injection)):
        batch_JSD = JSD_time_injection[i]

        for fpr in range(len(FPRs)):
            if np.max(batch_JSD) > thresholds[fpr]: 
                gw_pred[fpr].append(1)
            else: 
                gw_pred[fpr].append(0)

    # Calculate corresponding TPR
    
    TPRs = [float(np.sum(gw_pred[fpr]))/n_injection_events for fpr in range(len(FPRs))]
    index_FPR_1percent = 0 
    for i in FPRs: 
        if i < 0.01: 
            index_FPR_1percent+=1
    print('TPR for classifier at FPR=0.01: %s'%(TPRs[index_FPR_1percent]))
    
    index_FPR_10percent = 0 
    for i in FPRs: 
        if i < 0.1: 
            index_FPR_10percent+=1
    print('TPR for classifier at FPR=0.1: %s'%(TPRs[index_FPR_10percent]))
    
    return(TPRs, FPRs)


def clean(X_train_l1, X_train_h1, maximum):
    l1 = np.array([X_train_l1[i] for i in range(len(X_train_l1)) if X_train_l1[i].max() < maximum and X_train_h1[i].max() < maximum])
    h1 = np.array([X_train_h1[i] for i in range(len(X_train_h1)) if X_train_l1[i].max() < maximum and X_train_h1[i].max() < maximum])
    return(l1, h1)

def main(args):
    """ Main function to evaluate the model """
    global outdir
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s' % outdir)

    #load_H1 = h5.File('../../dataset/default_BNS_%s_randomtime.h5' % 'H1', 'r')
    #load_scaled_H1 = h5.File('../../dataset/default_BNS_%s_randomtime_scaled.h5' % 'H1', 'r')
    #load_L1 = h5.File('../../dataset/default_BNS_%s_randomtime.h5' % 'L1', 'r')
    #load_scaled_L1 = h5.File('../../dataset/default_BNS_%s_randomtime_scaled.h5' % 'L1', 'r')
    #load_SNR = h5.File('../ggwd/output/'+'default_BNS_8sec_14seed.hdf', 'r')
    
    load_noise = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_noise.hdf', 'r')
    load_signal = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_signal.hdf', 'r')
    
    #load_scaled = h5.File(f'../../dataset/default_BBH_{detector}.h5', 'r')
    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')
    
    datapoints = 2000

    ##### Evaluate Unsupervised methods ######
    
    #X_SNR = load_SNR['injection_parameters']['injection_snr'][:datapoints]
    #X_test_L1 = load_scaled_L1['injection'][:datapoints, 10000:12501]
    #X_train_L1= load_scaled_L1['noise'][:datapoints, 10000:12501]
    #X_test_H1 = load_scaled_H1['injection'][:datapoints, 10000:12501]
    #X_train_H1= load_scaled_H1['noise'][:datapoints, 10000:12501]
    
    global clip
    global badremoved
    
    clip = False
    badremoved = True
    
    if clip: 
        X_test_L1 = np.clip(load_signal['injection_samples']['l1_strain'][:datapoints, :16301], -150, 150)
        X_train_L1 = np.clip(load_noise['noise_samples']['l1_strain'][:datapoints, :16301], -150, 150)
        X_test_H1 = np.clip(load_signal['injection_samples']['h1_strain'][:datapoints, :16301], -150, 150)
        X_train_H1 = np.clip(load_noise['noise_samples']['h1_strain'][:datapoints, :16301], -150, 150)
        scaler_l1 = joblib.load('standard_scaler_l1_clip')
        scaler_h1 = joblib.load('standard_scaler_h1_clip')
    
    elif badremoved: 
        X_test_L1, X_test_H1 = clean(load_signal['injection_samples']['l1_strain'][:datapoints, 8704:13825],
                                     load_signal['injection_samples']['h1_strain'][:datapoints, 8704:13825],  150)
        X_train_L1, X_train_H1 = clean(load_noise['noise_samples']['l1_strain'][:datapoints, 8704:13825], 
                                       load_noise['noise_samples']['h1_strain'][:datapoints, 8704:13825], 150)
        scaler_l1 = joblib.load('standard_scaler_l1_badremoved')
        scaler_h1 = joblib.load('standard_scaler_h1_badremoved')
        print(X_test_L1.shape)
        
    X_test_L1 = scaler_l1.transform(X_test_L1.reshape((-1, 1))).reshape((-1, 13825-8704))
    X_train_L1 = scaler_l1.transform(X_train_L1.reshape((-1, 1))).reshape((-1, 13825-8704))
    X_test_H1 = scaler_h1.transform(X_test_H1.reshape((-1, 1))).reshape((-1, 13825-8704))
    X_train_H1 = scaler_h1.transform(X_train_H1.reshape((-1, 1))).reshape((-1, 13825-8704))
        
    #del load_scaled_L1, load_scaled_H1, load_L1, load_H1
    
    #directory_list = [outdir]
    #names = ['LSTM Autoencoder']
    #timesteps = [timesteps]
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_unsupervised_GRU_100', 'BBH_training_supervised_PaperConv_BBH_2']#, 'BIGsimdata_L1_2KHz_unsupervised_filtered_DNN' ]#, 'BIGsimdata_L1_2KHz_unsupervised_filtered_ConvDNN', 'BIGsimdata_L1_2KHz_unsupervised_filtered_LSTM']
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_supervsed_LSTM_tanh_BBH']
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_unsupervised_GRU_100', 'BBH_training_supervised_PaperConv_BBH_2']
    #directory_list = ['BBH_training_supervsed_LSTM_tanh_BNS', 'BNS_training_unsupervised_GRU_100', 'BNS_training_unsupervised_PaperConv']
    directory_list = ['LSTM']#, 'GRU', 'CNN']
    
    
    names_unsupervised = ['LSTM Autoencoder']#, 'GRU Autoencoder', 'CNN Autoencoder']#, 'DNN Autoencoder']#, 'CNN-DNN Autoencoder', 'LSTM Autoencoder']
    timesteps = [100, 100, 1024]
    FPR_set = []
    TPR_set = []
    
    for name, directory, timestep in zip(names_unsupervised, directory_list, timesteps): 
        print('Determining performance for: %s'%(name))
        TPR, FPR = TPR_FPR_arrays_latentspace(X_train_L1[:, :-int(np.shape(X_train_L1)[1]%timestep)], 
                                                X_test_L1[:, :-int(np.shape(X_test_L1)[1]%timestep)], 
                                                 X_train_H1[:, :-int(np.shape(X_train_H1)[1]%timestep)], 
                                                 X_test_H1[:, :-int(np.shape(X_test_H1)[1]%timestep)], directory, timestep)
        #TPR, FPR = TPR_FPR_arrays(X_train_L1[:, :-int(np.shape(X_train_L1)[1]%timestep)], X_test_L1[:, :-int(np.shape(X_test_L1)[1]%timestep)], directory, timestep)
        #TPR, FPR = TPR_FPR_arrays(X_train[:, 7500:13500], X_test[:, 7500:13500], directory, timestep)
        TPR_set.append(TPR)
        FPR_set.append(FPR)
        print('Done!')
    
    ##### Plotting both methods ######
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(10, 10))
    lw = 2
    
    '''
    for name, directory, pred in zip(names_supervised, directory_list, predictions): 
        print('Determining performance for: %s'%(name))
        print(np.shape(test_truth[:, 1]))
        print(pred)
        fpr, tpr, thresholds = roc_curve(test_truth[:, 1], pred[:, 1])
        
        index_FPR_1percent = 0 
        for i in fpr: 
            if i < 0.01: 
                index_FPR_1percent+=1
        print('TPR for classifier at FPR=0.01: %s'%(tpr[index_FPR_1percent]))

        index_FPR_10percent = 0 
        for i in fpr: 
            if i < 0.0001: 
                index_FPR_10percent+=1
        print('TPR for classifier at FPR=0.0001: %s'%(tpr[index_FPR_10percent]))
        
        ax.plot(fpr, tpr, lw=2, label='%s (auc = %0.2f)'%(name, auc(fpr, tpr)))
        print('Accuracy: %s'%(accuracy_score(np.argmax(test_truth, axis=-1), np.argmax(pred, axis=-1))))
        print('Done!')
    '''
    
    for FPRs, TPRs, name in zip(FPR_set, TPR_set, names_unsupervised):
        ax.plot(FPRs, TPRs,
             lw=lw, label='%s (auc = %0.2f)'%(name, auc(FPRs, TPRs)))
    import matplotlib.ticker as plticker
    ax.xaxis.set_major_locator(plticker.MultipleLocator(base=0.1))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(base=0.02))
    ax.tick_params(direction='in', axis='both', which='major', labelsize=12, length=12 )
    ax.tick_params(direction='in', axis='both', which='minor' , length=6)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')    
    ax.semilogx()
    ax.grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
    ax.grid(which='major', alpha=0.9, linestyle='dotted')
    leg = ax.legend(borderpad=1, frameon=False, loc=2, fontsize=14)
    leg._legend_box.align = "left"
    
    ax.set_xlim([-3, 1])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    #plt.xscale('log')
    ax.set_title('LIGO Latent-Space BBH Detection')
    #sf.legend(loc="upper left", fontsize=9)
    f.savefig('%s/ROC_curve_log_BBHdataset_5e-5_400step_temp.jpg'%(outdir))
    
    ### Enable if needed - these are additional plots to check if methods are working in unsupervised learning approach###
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt 
    import matplotlib.ticker as plticker
    
    
    X_test_graph = []
    for i in range(len(1000)): 
        X_test_graph.append([X_test_L1[i], X_test_H1[i]])
            
    print(np.shape(X_test_graph))
    for n in range(20):
        
        f, ax = plt.subplots(figsize=(10, 6), nrows=2, ncols=1)
        ax[0].xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        #ax[0].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
        ax[0].tick_params(direction='in', axis='both', which='major', labelsize=10, length=12 )
        ax[0].tick_params(direction='in', axis='both', which='minor' , length=6)
        ax[0].xaxis.set_ticks_position('both')
        ax[0].yaxis.set_ticks_position('both')
        ax[0].set_xticklabels([])
        ax[0].grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax[0].grid(which='major', alpha=0.9, linestyle='dotted')
        leg0 = ax[0].legend(borderpad=1, frameon=False, loc=2, fontsize=10)
        leg0._legend_box.align = "left"
        ax[1].xaxis.set_major_locator(plticker.MultipleLocator(base=0.5))
        #ax[1].xaxis.set_minor_locator(plticker.MultipleLocator(base=0.5))
        ax[1].tick_params(direction='in', axis='both', which='major', labelsize=10, length=12 )
        ax[1].tick_params(direction='in', axis='both', which='minor' , length=6)
        ax[1].xaxis.set_ticks_position('both')
        ax[1].yaxis.set_ticks_position('both')    
        ax[1].grid(which='minor', alpha=0.5, axis='y', linestyle='dotted')
        ax[1].grid(which='major', alpha=0.9, linestyle='dotted')
        leg1 = ax[1].legend(borderpad=1, frameon=False, loc=2)
        leg1._legend_box.align = "left"
        ax[0].set_xlim(0, 8)
        ax[1].set_xlim(0, 8)
        ax[0].set_ylim(0, 0.07)
        ax[1].set_ylim(0, 0.07)
        
        random_samples = RandSample(range(0, len(X_test_graph)), 2)
        for num, random_sample in zip(range(len(random_samples)), random_samples): 
            event_L1 = np.array(X_test_graph)[random_sample, 0, :16300]
            event_H1 = np.array(X_test_graph)[random_sample, 1, :16300]
            print(np.shape(event_L1))
            print(np.shape(event_H1))
            event_L1 = event_L1.reshape(-1, int(args.timesteps), 1)
            event_H1 = event_H1.reshape(-1, int(args.timesteps), 1)
            time = random_sample
            loss_fn = MeanSquaredError(reduction='none')

            #model = load_model('%s/best_model_H1.hdf5'%(outdir))
            model = load_model('../snnTorch/LSTM')
            X_pred_L1 = model.predict(event_L1)
            X_pred_H1 = model.predict(event_H1)
            #threshold_0.1 = threshold_given_FPR(X_train[:, :-int(np.shape(X_train)[1]%timestep)], X_test[:, :-int(np.shape(X_test)[1]%timestep)], directory, timestep, FPR=0.1)
            timing = [i*100/2048. for i in range(163)]
            losses_L1 = loss_fn(event_L1, X_pred_L1).numpy()
            losses_H1 = loss_fn(event_H1, X_pred_H1).numpy()
            batch_loss_L1 = np.mean(losses_L1, axis=1)
            batch_loss_H1 = np.mean(losses_H1, axis=1)
            batch_loss = [h1 + l1 for h1, l1 in zip(batch_loss_H1, batch_loss_L1)]
            
            ax[num].plot(timing, batch_loss,  label='LSTM Autoencoder')
            ax[num].axvline(5.5, label='GW Peak Intensity', color='green')
            leg0 = ax[0].legend(borderpad=1, frameon=False, loc=2, fontsize=16)
            leg0._legend_box.align = "left"
            
            ax[0].yaxis.set_major_locator(plticker.MultipleLocator(base=0.01))
            
            ax[1].yaxis.set_major_locator(plticker.MultipleLocator(base=0.01))
            
            #plt.xlabel('Timestep')
            #plt.ylabel('Loss')
            #plt.title('LSTM Autoencoder Output')
            #plt.axhline(threshold, label='GW event threshold', color='red')
            #plt.legend(loc='upper left')

        ax[1].set_xlabel('Time (seconds)')
        ax[0].set_ylabel('Loss')
        ax[1].set_ylabel('Loss')
        ax[0].set_title('LSTM Autoencoder Output')
        plt.savefig('%s/loss/batchloss_%s.jpg'%(outdir,time))
        
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
