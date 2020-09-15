""" Train autoencoder for anomaly detection in given time series data. """

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_unsupervised import autoencoder_LSTM, autoencoder_ConvLSTM, autoencoder_ConvDNN, autoencoder_DNN, \
    autoencoder_Conv, autoencoder_Conv2


def hls4ml_deployment(model, test_data, test_truth):
    import hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='model', default_precision='f32')
    print(config)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='oneAPI', device="cpu",
                                                           output_dir='test')
    hls_model.compile(batch_size=1)
    y_hls = hls_model.predict(test_data.astype(np.float32))
    # print(f"Y hls: {y_hls}")
    from sklearn.metrics import accuracy_score
    print("Keras  Accuracy: {}".format(
        accuracy_score(np.argmax(test_truth, axis=1), np.argmax(model.predict(test_data), axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(test_truth, axis=1), np.argmax(y_hls, axis=1))))


def main():
    """ Main function to prepare and train the model """
    outdir = 'oneAPI-model'
    os.system('mkdir -p %s' % outdir)

    load = h5.File('data.h5', 'r')

    datapoints = 1200
    X_train = load['data'][:]

    # Reshape inputs for LSTM [samples, timesteps, features]
    X_train = X_train.reshape(-1, datapoints, 1)
    print("Training data shape:", X_train.shape)

    # Define the model
    model = autoencoder_DNN(X_train)
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    # Fit the model to the data
    nb_epochs = 1
    batch_size = 1024
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint(outdir + '/best_model.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(X_train, X_train, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[early_stop, mcp_save]).history
    model.save(outdir + '/last_model.hdf5')

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(outdir + '/loss.png')

    hls4ml_deployment(model, X_train, X_train)


if __name__ == "__main__":
    main()
