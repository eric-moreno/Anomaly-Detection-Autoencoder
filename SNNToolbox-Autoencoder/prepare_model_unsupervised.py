""" Train autoencoder for anomaly detection in given time series data. """

import os
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
from sklearn.preprocessing import MinMaxScaler
from gwpy.timeseries import TimeSeries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from architectures_unsupervised import autoencoder_ConvDNN, autoencoder_DNN, autoencoder_Conv

sns.set(color_codes=True)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def prepare_model():
    """ Main function to prepare and train the model """
    outdir = "Outputs"
    detector = "L1"
    freq = 2
    filtered = 1
    timesteps = 100
    os.system(f'mkdir {outdir}')

    # Load train and test data
    load = h5.File('../../dataset/default_simulated.hdf', 'r')

    # Define frequency in Hz instead of KHz
    if int(freq) == 2:
        freq = 2048
    elif int(freq) == 4:
        freq = 4096
    else:
        return print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

    noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:][:]
    print("Noise samples shape:", noise_samples.shape)
    injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:]
    print("Injection samples shape:", injection_samples.shape)

    # With LIGO simulated data, the sample isn't pre-filtered so need to filter again.
    # Real data is not filtered yet.
    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print(f'Sample Frequency: {freq} Hz')
        x = [filters(sample, freq) for sample in noise_samples]
        print('Filtering completed')

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(x)
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    joblib.dump(scaler, scaler_filename)

    # Trim dataset to be batch-friendly and reshape into timestep format
    x = []
    for event in range(len(X_train)):
        if X_train[event].shape[0] % timesteps != 0:
            x.append(X_train[event][:-1 * int(X_train[event].shape[0] % timesteps)])
    X_train = np.array(x)

    x = []
    X_test = injection_samples
    for event in range(len(X_test)):
        if X_test[event].shape[0] % timesteps != 0:
            x.append(X_test[event][:-1 * int(X_test[event].shape[0] % timesteps)])
    X_test = np.array(x)

    # Reshape inputs for LSTM
    X_train = X_train.reshape(-1, timesteps)
    print("Training data shape:", X_train.shape)
    np.savez('x_test.npz', arr_0=X_train)
    X_test = X_test.reshape(-1, timesteps)
    print("Test data shape:", X_test.shape)
    np.savez('y_test.npz', arr_0=X_train)
    print("Test and Train data saved in npz format")

    # Define the model
    model = autoencoder_DNN(X_train)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Fit the model to the data
    nb_epochs = 200
    batch_size = 1024
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(f'{outdir}/best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[early_stop, mcp_save]).history
    model.save(f'{outdir}/last_model.hdf5')

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(f'{outdir}/loss.jpg')


if __name__ == "__main__":
    prepare_model()
