""" Anomaly Detection with Spiking Neural Networks deployed on the Loihi chip. """

import os
import h5py as h5
import nengo_dl
import tensorflow as tf
import joblib
from gwpy.timeseries import TimeSeries
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

load = h5.File('../../dataset/default_simulated.hdf', 'r')
freq = 2048
detector = 'L1'
filtered = 1
timesteps = 100
outdir = 'training_autoencoder'
os.system('mkdir -p %s' % outdir)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:][:]

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
if X_train.shape[0] % timesteps != 0:
    X_train = X_train[:-1 * int(X_train.shape[0] % timesteps)]

# Reshape inputs for LSTM [samples, timesteps, features]
X = X_train.reshape(-1, timesteps, 1)
print("Training data shape:", X_train.shape)

# Define the model
inputs = Input(shape=(X.shape[1], X.shape[2]))
x = Flatten()(inputs)
x = Dense(int(X.shape[1] / 2), activation='relu')(x)
x = Dense(int(X.shape[1] / 10), activation='relu')(x)
x = Dense(int(X.shape[1] / 2), activation='relu')(x)
x = Dense(X.shape[1], activation='relu')(x)
output = Reshape((X.shape[1], 1))(x)
model = Model(inputs=inputs, outputs=output)

# Convert the model to nengo_dl format
converter = nengo_dl.Converter(model)

with nengo_dl.Simulator(converter.net, minibatch_size=200) as sim:
    # run training
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.metrics.sparse_categorical_accuracy],
    )

    sim.fit(
        {converter.inputs[inputs]: X},
        {converter.outputs[output]: X},
        validation_data=(
            {converter.inputs[inputs]: X},
            {converter.outputs[output]: X},
        ),
        epochs=2,
    )

    # save the parameters to file
    sim.save_params("./keras_to_snn_params")
