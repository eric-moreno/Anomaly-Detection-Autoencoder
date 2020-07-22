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

sns.set(color_codes=True)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def main(args):
    """ Main function to evaluate the model """
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s' % outdir)

    load = h5.File('../../dataset/default_simulated.hdf', 'r')

    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

    n_noise_events = 1000
    noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:][:n_noise_events]

    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print(f'Sample Frequency: {freq} Hz')
        x_noise = [filters(sample, freq) for sample in noise_samples]
        print('Filtering completed')

    # Load previous scaler and transform
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    scaler = joblib.load(scaler_filename)
    X_train = scaler.transform(x_noise)

    # Trim dataset to be batch-friendly
    # if X_train.shape[0]%timesteps != 0:
    #    X_train = X_train[:-1*int(X_train.shape[0]%timesteps)]

    # Reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(-1, timesteps, 1)
    print("Training data shape:", X_train.shape)

    # Load the autoencoder network model
    model = load_model(f'{outdir}/best_model.hdf5')

    # Evaluating on training data to find threshold
    print('Evaluating Model on train data. This make take a while...')
    X_pred = model.predict(X_train)
    print('Finished evaluating model on train data')

    loss_fn = MeanSquaredError(reduction='none')
    losses = loss_fn(X_train, X_pred).eval(session=tf.compat.v1.Session())
    averaged_losses = np.mean(losses, axis=1)

    '''
    X_pred = X_pred.reshape(X_pred.shape[0]*timesteps, X_pred.shape[2])
    X_pred = pd.DataFrame(X_pred)
    scored = pd.DataFrame()
    Xtrain = X_train.reshape(X_train.shape[0]*timesteps, X_train.shape[2])
    scored['Loss_mae'] = np.mean(np.abs(X_pred-Xtrain), axis = 1)
    plt.figure(figsize=(16,9), dpi=80)
    plt.title('Loss Distribution', fontsize=16)
    sns.distplot(scored['Loss_mae'], bins = 20, kde= True, color = 'blue');
    plt.savefig('%s/loss_train_spread.jpg'%(outdir))
    '''

    threshold = np.max(averaged_losses)
    print(f'The threshold is: {threshold}')

    roc_steps = 20
    FPRs = np.logspace(-2, 1.9999999, roc_steps)
    thresholds = []
    for fpr in FPRs:
        thresholds.append(np.percentile(averaged_losses, fpr))

    # Evaluate on 10 test data events
    n_injection_events = 1000
    injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:][:n_injection_events]
    times = load['injection_samples']['event_time']
    random_samples = RandSample(range(0, len(injection_samples)), 10)

    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        x_injection = [filters(sample, freq) for sample in injection_samples]
        print('Filtering completed')

    # Normalize the data
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    scaler = joblib.load(scaler_filename)
    X_test = scaler.transform(x_injection)
    # X_test = scaler.transform(y.reshape(-1, 1))

    gw_pred = [[] for i in range(roc_steps)]
    for i in range(len(X_test)):
        event = X_test[i]
        if event.shape[0] % timesteps != 0:
            event = event[:-1 * int(event.shape[0] % timesteps)]
        event = event.reshape(-1, timesteps, 1)

        X_pred = model.predict(event)

        loss_fn = MeanSquaredError(reduction='none')
        losses = loss_fn(event, X_pred).eval(session=tf.compat.v1.Session())
        batch_loss = np.mean(losses, axis=1)

        for fpr in range(len(FPRs)):
            if np.max(batch_loss) >= thresholds[fpr]:
                gw_pred[fpr].append(1)
            else:
                gw_pred[fpr].append(0)

    FPRs = np.array(FPRs) / 100
    TPRs = []
    for fpr in range(len(FPRs)):
        TP = np.sum(gw_pred[fpr])
        TPRs.append(TP / float(n_injection_events))

    # print('Model has correctly identified %s gravitational-waves'%())
    # fpr, tpr, _ = roc_curve(gw_truth, gw_pred)

    roc_auc = auc(FPRs, TPRs)
    plt.figure()
    lw = 2
    plt.plot(FPRs, TPRs, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([1e-4, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('LIGO anomaly detection algorithm LSTM')
    plt.legend(loc="lower right")
    plt.savefig(f'{outdir}/ROC_curve.jpg')
    sys.exit()

    for random_sample in random_samples:
        event = X_test[random_sample]
        time = times[random_sample] - 1000000000

        if event.shape[0] % timesteps != 0:
            event = event[:-1 * int(event.shape[0] % timesteps)]
        event = event.reshape(-1, timesteps, 1)

        X_pred = model.predict(event)

        loss_fn = MeanSquaredError(reduction='none')
        losses = loss_fn(event, X_pred).eval(session=tf.compat.v1.Session())
        batch_loss = np.mean(losses, axis=1)

        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(batch_loss)
        plt.axvline(len(batch_loss) * 5.5 / 8, label='actual GW event', color='green')
        plt.axhline(threshold, label='GW event threshold', color='red')
        plt.legend(loc='upper left')
        plt.savefig(f'{outdir}/batchloss_{time}.jpg')

        X_pred_test = np.array(model.predict(event))

        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(event.reshape(-1)[int(2048 * 5.5) - 300:int(2048 * 5.5) + 300], label='truth')
        ax.plot(X_pred_test.reshape(-1)[int(2048 * 5.5) - 300:int(2048 * 5.5) + 300], label='predict')
        plt.legend(loc='upper left')
        plt.title('LSTM Autoencoder')
        plt.savefig(f'{outdir}/middle30ms_{time}.jpg')

        print(X_pred_test.shape)
        X_pred_test = X_pred_test.reshape(X_pred_test.shape[0] * timesteps, X_pred_test.shape[2])

        # X_pred_train.index = train.index
        Xtest = event.reshape(event.shape[0] * timesteps, event.shape[2])

        X_pred_test = pd.DataFrame(X_pred_test)
        scored_test = pd.DataFrame()
        scored_test['Loss_mae'] = np.mean(np.abs(X_pred_test - Xtest), axis=1)
        # scored_test['Threshold'] = threshold
        # scored_test['Anomaly'] = scored_test['Loss_mae'] > scored_test['Threshold']
        # scored_test.plot(logy=True,  figsize=(16,9), ylim=[t/(1e2),threshold*(1e2)], color=['blue','red'])
        scored_test.plot(logy=False, figsize=(16, 9), color=['blue', 'red'])
        # Sampling rate of 2048 Hz with the event occuring 5.5 seconds into sample
        plt.axvline(5.5 * 2048, label='actual GW event', color='green')
        plt.legend(loc='upper left')
        plt.savefig('%s/test_threshold_%s_8sec.jpg' % (outdir, time))


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
