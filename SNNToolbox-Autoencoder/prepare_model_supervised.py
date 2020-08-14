""" Train autoencoder for anomaly detection in given time series data. """

import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
from gwpy.timeseries import TimeSeries
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils.multiclass import unique_labels
from architectures_supervised import autoencoder_ConvDNN, autoencoder_DNN

sns.set(color_codes=True)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix. Normalization can be added by setting `normalize=True` """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(title)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def prepare_model():
    """ Main function to prepare and train the model """
    outdir = "Outputs"
    os.system(f'mkdir {outdir}')

    # Load train and test data
    load = h5.File('../../dataset/240k_1sec_L1.h5', 'r')
    X_train = load['data'][:]

    datapoints = 120000
    gw = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    noise = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    targets = np.transpose(np.array([gw, noise]))

    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(X_train, targets, test_size=0.2, random_state=42)
    class_names = np.array(['noise', 'GW'], dtype=str)

    print("Train data shape:", train_data.shape)
    print("Train labels data shape:", train_truth.shape)
    print("Test data shape:", test_data.shape)
    print("Test labels data shape:", test_truth.shape)

    np.savez('x_test.npz', arr_0=test_data)
    np.savez('y_test.npz', arr_0=test_truth)
    print("Test and Train data saved in npz format")

    # Define a model
    model = autoencoder_ConvDNN(train_data)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    model.summary()

    # Fit the model to the data
    nb_epochs = 10
    batch_size = 16
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
    mcp_save = ModelCheckpoint('autoencoder2SNN.h5', save_best_only=True, monitor='val_loss', mode='min')
    history = model.fit(train_data, train_truth, epochs=nb_epochs, batch_size=batch_size,
                        validation_split=0.2, callbacks=[earlyStopping, mcp_save]).history
    model.save(f'{outdir}/last_model.h5')

    fig, ax = plt.subplots(figsize=(14, 6), dpi=80)
    ax.plot(history['loss'], 'b', label='Train', linewidth=2)
    ax.plot(history['val_loss'], 'r', label='Validation', linewidth=2)
    ax.set_title('Model loss', fontsize=16)
    ax.set_ylabel('Loss (mse)')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    plt.savefig(f'{outdir}/loss.jpg')

    # Evaluate the model
    predictions = model.predict(test_data)

    # Generate a ROC curve
    plt.figure()
    fpr, tpr, thresholds = roc_curve(test_truth.argmax(axis=1), predictions.argmax(axis=1))
    plt.plot(fpr, tpr, lw=2, label='%s (auc = %0.2f)' % ('ANN model for SNN-TB', auc(fpr, tpr)))
    plt.xlim([1e-4, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('LIGO Supervised GW-Detection')
    plt.legend(loc="lower right")
    plt.savefig('%s/ROC_curve.jpg' % outdir)

    # Generate unnormalized confusion matrix
    plot_confusion_matrix(test_truth.argmax(axis=1), predictions.argmax(axis=1), classes=class_names,
                          title='Confusion matrix, without normalization')
    plt.savefig(outdir + '/confusion_matrix_unnormalized.jpg')

    # Generate normalized confusion matrix
    plot_confusion_matrix(test_truth.argmax(axis=1), predictions.argmax(axis=1), classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(outdir + '/confusion_matrix_normalized.jpg')


if __name__ == "__main__":
    prepare_model()
