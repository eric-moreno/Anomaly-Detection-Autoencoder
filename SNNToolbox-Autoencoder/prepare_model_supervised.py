""" Train autoencoder for anomaly detection in given time series data. """

import os
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from gwpy.timeseries import TimeSeries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from architectures_supervised import autoencoder_ConvDNN, autoencoder_DNN

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

    datapoints = 5000
    noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:datapoints]
    injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:datapoints]
    train_data = np.concatenate((noise_samples, injection_samples))

    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))

    train_truth = np.transpose(np.array([gw, noise]))
    train_data, train_truth = shuffle(train_data, train_truth)

    # With LIGO simulated data, the sample isn't pre-filtered so need to filter again.
    # Real data is not filtered yet.
    if bool(int(filtered)):
        print('Filtering data with whitening and bandpass')
        print(f'Sample Frequency: {freq} Hz')
        x = [filters(sample, freq)[7168:15360] for sample in train_data]
        print('Filtering completed')

    # Normalize the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(x)
    scaler_filename = f"{outdir}/scaler_data_{detector}"
    joblib.dump(scaler, scaler_filename)

    print("Training data shape:", X_train.shape)
    print("Testing data shape:", train_truth.shape)

    np.savez('x_test.npz', arr_0=X_train)
    np.savez('y_test.npz', arr_0=train_truth)
    print("Test and Train data saved in npz format")

    # Define model
    model = autoencoder_DNN(X_train)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # fit the model to the data
    nb_epochs = 300
    batch_size = 16
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(f'{outdir}/best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, train_truth, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
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
