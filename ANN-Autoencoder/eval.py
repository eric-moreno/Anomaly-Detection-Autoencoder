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
from sklearn.metrics import auc
import setGPU 
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
        
        divide_into = 2
        
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
    model = load_model('%s/best_model_%s.hdf5'%(model_outdir, args.detector))
    #model = load_model('%s/best_model.hdf5'%(model_outdir))
    
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

def TPR_FPR_arrays_doubledetector(noise_array_L1, injection_array_L1, noise_array_H1, injection_array_H1, model_outdir, steps, num_entries=10): 
    # load the autoencoder network model
    model_L1 = load_model('%s/best_model_L1.hdf5'%(model_outdir))
    model_H1 = load_model('%s/best_model_H1.hdf5'%(model_outdir))
    
    n_noise_events = np.shape(noise_array_L1)[0]
    n_injection_events = np.shape(injection_array_L1)[0]
    
    noise_array_L1 = noise_array_L1.reshape(-1, steps, 1)
    injection_array_L1 = injection_array_L1.reshape(-1, steps, 1)
    noise_array_H1 = noise_array_H1.reshape(-1, steps, 1)
    injection_array_H1 = injection_array_H1.reshape(-1, steps, 1)
    
    ### Evaluating on training data to find threshold ### 
    print('Evaluating Model on train data. This make take a while...')
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
    
    max_loss_L1 = [np.max(event_L1) for event_L1 in averaged_losses_L1]
    max_loss_H1 = [np.max(event_H1) for event_H1 in averaged_losses_H1]
    
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

def evaluate_supervised_methods(data, datapoints, detec, outdir): 
    #### Evaluate Supervised methods ######
    
    load = data
    
    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    targets = np.transpose(np.array([gw, noise]))
    
    X = np.concatenate((load['injection'][:datapoints, :], load['noise'][:datapoints, :]))
    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(X, targets, test_size=0.2, random_state=42)
    class_names = np.array(['noise', 'GW'], dtype=str)

    print(train_data.shape)
    # Reshape inputs
    train_data = train_data.reshape((train_data.shape[0], -1))
    print("Train data shape:", train_data.shape)
    #train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
    print("Train labels data shape:", train_truth.shape)
    test_data = test_data.reshape((test_data.shape[0], -1))
    print("Test data shape:", test_data.shape)
    #test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
    print("Test labels data shape:", test_truth.shape)
    
    print('Evaluating supervised learning models')
    model = load_model('%s/best_model_%s.hdf5'%(outdir, detec))
    model_alt = load_model('%s/best_model_%s.hdf5'%('BNS_training_supervised_ConvDNN', detec))
    
    X_pred_test_native = model.predict(test_data)
    X_pred_test_alt = model_alt.predict(test_data)
    
    del X, train_data, test_data
    predictions = [X_pred_test_native, X_pred_test_alt]
    
    return(predictions)
def main(args):
    """ Main function to evaluate the model """
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s' % outdir)
    
    load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_12seed.hdf', 'r')
    
    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')
    
    datapoints = 25000

    ##### Evaluate Unsupervised methods ######
    
    scaler = joblib.load('../scalers/standard_scaler') 
    X_test_L1 = scaler.transform(load['injection_samples']['l1_strain'][:datapoints, 8704:13825].reshape((-1, 1))).reshape((-1,13825-8704))
    X_train_L1 = scaler.transform(load['noise_samples']['l1_strain'][:datapoints, 8704:13825].reshape((-1, 1))).reshape((-1,13825-8704))
    X_test_H1 = scaler.transform(load['injection_samples']['h1_strain'][:datapoints, 8704:13825].reshape((-1, 1))).reshape((-1,13825-8704))
    X_train_H1 = scaler.transform(load['noise_samples']['h1_strain'][:datapoints, 8704:13825].reshape((-1, 1))).reshape((-1,13825-8704))
        
    #del load_scaled_L1, load_scaled_H1, load_L1, load_H1
    
    ###### Names of old directories - just keeping to remember where things were. Can delete if needed...#####
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_unsupervised_GRU_100', 'BBH_training_supervised_PaperConv_BBH_2']#, 'BIGsimdata_L1_2KHz_unsupervised_filtered_DNN' ]#, 'BIGsimdata_L1_2KHz_unsupervised_filtered_ConvDNN', 'BIGsimdata_L1_2KHz_unsupervised_filtered_LSTM']
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_supervsed_LSTM_tanh_BBH']
    #directory_list = ['BBH_training_unsupervsed_tanhLSTM', 'BBH_training_unsupervised_GRU_100', 'BBH_training_supervised_PaperConv_BBH_2']
    #directory_list = ['BBH_training_supervsed_LSTM_tanh_BNS', 'BNS_training_unsupervised_GRU_100', 'BNS_training_unsupervised_PaperConv']
    
    ### List 1+ unsupervised training directories here ###
    directory_list = ['LSTM', 'GRU', 'CNN'] # will look inside these directories
    names_unsupervised = ['LSTM Autoencoder ', 'GRU Autoencoder', 'CNN Autoencoder'] # plot with these names
    timesteps = [100, 100, 1024] # use these timesteps for the corresponding above models

    
    FPR_set = []
    TPR_set = []
    
    for name, directory, timestep in zip(names_unsupervised, directory_list, timesteps): 
        print('Determining performance for: %s'%(name))
        
        # Determine performance for double detector
        TPR, FPR = TPR_FPR_arrays_doubledetector(X_train_L1[:, :-int(np.shape(X_train_L1)[1]%timestep)], 
                                                X_test_L1[:, :-int(np.shape(X_test_L1)[1]%timestep)], 
                                                 X_train_H1[:, :-int(np.shape(X_train_H1)[1]%timestep)], 
                                                 X_test_H1[:, :-int(np.shape(X_test_H1)[1]%timestep)], directory, timestep)
        
        # Determine performance for a single detector 
        #TPR, FPR = TPR_FPR_arrays(X_train_L1[:, :-int(np.shape(X_train_L1)[1]%timestep)], X_test_L1[:, :-int(np.shape(X_test_L1)[1]%timestep)], directory, timestep)
        TPR_set.append(TPR)
        FPR_set.append(FPR)
        print('Done!')
    
    
    ##### Plotting unsupervised methods ######
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(10, 10))
    lw = 2
    
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
    ax.set_title('LIGO Single-Detector BBH Detection')
    #sf.legend(loc="upper left", fontsize=9)
    f.savefig('%s/ROC_curve_log_BBHdataset_5e-5_400step_temp.jpg'%(outdir))

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
