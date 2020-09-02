import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py as h5
from keras.callbacks import EarlyStopping, ModelCheckpoint
from model_supervised import autoencoder_ConvDNN, autoencoder_DNN, autoencoder_ConvDNN_Nengo


def hls4ml_deployment(model, test_data, test_truth):
    import hls4ml
    config = hls4ml.utils.config_from_keras_model(model, granularity='model', default_precision='f32')
    print(config)
    hls_model = hls4ml.converters.convert_from_keras_model(model, hls_config=config, backend='oneAPI', device="cpu",
                                                           output_dir='test')
    hls_model.compile(batch_size=1)
    y_hls = hls_model.predict(test_data.astype(np.float32))
    print(f"Y hls: {y_hls}")
    from sklearn.metrics import accuracy_score
    print("Keras  Accuracy: {}".format(
        accuracy_score(np.argmax(test_truth, axis=1), np.argmax(model.predict(test_data), axis=1))))
    print("hls4ml Accuracy: {}".format(accuracy_score(np.argmax(test_truth, axis=1), np.argmax(y_hls, axis=1))))


def main():
    outdir = 'oneAPI-model'
    os.system('mkdir -p %s' % outdir)

    load = h5.File('data.h5', 'r')

    datapoints = 1200
    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    targets = np.transpose(np.array([gw, noise]))

    X = load['data'][:]
    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(X, targets, test_size=0.2, random_state=42)

    print(train_data.shape)
    # Reshape inputs for autoencoder_DNN and autoencoder_ConvDNN
    # train_data = train_data.reshape((train_data.shape[0], 1, -1))
    # train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
    # test_data = test_data.reshape((test_data.shape[0], 1, -1))
    # test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
    print("Train data shape:", train_data.shape)
    print("Train labels data shape:", train_truth.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels data shape:", test_truth.shape)

    # Define model
    model = autoencoder_DNN(train_data)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # fit the model to the data
    nb_epochs = 300
    batch_size = 16
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('%s/best_model.hdf5' % outdir, save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(train_data, train_truth, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
    model.save('%s/last_model.hdf5' % outdir)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig('%s/loss.jpg' % outdir)

    hls4ml_deployment(model, test_data, test_truth)


if __name__ == "__main__":
    main()
