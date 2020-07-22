""" Evaluation of models trained for anomaly detection. """

import os
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
from sklearn.externals import joblib
from gwpy.timeseries import TimeSeries
from keras.models import load_model
from random import sample as RandSample

sns.set(color_codes=True)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def main(args):
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    timesteps = int(args.timesteps)
    os.system('mkdir -p %s' % outdir)

    load = h5.File('../../dataset/default.hdf', 'r')

    ''' #### WONT WORRY ABOUT THRESHOLDING FOR NOW
    noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:][:10]
    
    # Definining frequency in Hz instead of KHz
    #freq = 1000*int(freq)
    freq = 4000
    
    if bool(int(filtered)):
        print('filtering data with whitening and bandpass')
        x = [filters(sample, freq) for sample in noise_samples]
        print('Done!')
        
    # Load previous scaler and transform    
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    scaler = joblib.load(scaler_filename) 
    X_train = scaler.transform(x)
    
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
    '''

    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq} is not supported')

    # Evaluate model on 10 test data events
    injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:][:1000]
    times = load['injection_samples']['event_time']
    random_samples = RandSample(range(0, len(injection_samples)), 10)

    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print(f'Sample Frequency: {freq} Hz')
        x = [filters(sample, freq) for sample in injection_samples]
        print('Filtering completed')

    # Normalize the data
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    scaler = joblib.load(scaler_filename)
    X_test = scaler.transform(x)
    # X_test = scaler.transform(y.reshape(-1, 1))

    for random_sample in random_samples:
        event = X_test[random_sample]
        time = times[random_sample] - 1000000000

        if event.shape[0] % timesteps != 0:
            event = event[:-1 * int(event.shape[0] % timesteps)]
        event = event.reshape(-1, timesteps, 1)

        # load the autoencoder network model
        model = load_model(f'{outdir}/best_model.hdf5')

        batch_loss = []
        for batch in event:
            batch_loss.append(model.evaluate(batch.reshape(1, timesteps, 1), batch.reshape(1, timesteps, 1), verbose=0))

        fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
        ax.plot(batch_loss)
        plt.axvline(len(batch_loss) * 5.5 / 8, label='actual GW event', color='green')
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
        plt.savefig(f'{outdir}/test_threshold_{time}_8sec.jpg')


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
