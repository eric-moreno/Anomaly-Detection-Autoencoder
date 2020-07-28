""" Autoencoder models convertable to SNNs for anomaly detection. """

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Reshape
from keras.models import Model


def autoencoder_ConvDNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Reshape((X.shape[1], 1))(inputs)
    L1 = Conv1D(64, 16, activation="relu", dilation_rate=1)(x)
    L2 = MaxPooling1D(4)(L1)
    L3 = Conv1D(128, 16, activation="relu", dilation_rate=2)(L2)
    L4 = MaxPooling1D(4)(L3)
    L5 = Conv1D(256, 16, activation="relu", dilation_rate=2)(L4)
    L6 = MaxPooling1D(4)(L5)
    L7 = Conv1D(512, 32, activation="relu", dilation_rate=2)(L6)
    L8 = MaxPooling1D(4)(L7)
    x = Flatten()(L8)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model
