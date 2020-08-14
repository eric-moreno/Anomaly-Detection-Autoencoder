""" Autoencoder models convertable to SNNs for anomaly detection. """

from keras.layers import Input, Dense, Conv2D, Flatten, Reshape
from keras.models import Model


def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def autoencoder_ConvDNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Reshape((X.shape[1], 1, 1))(inputs)
    x = Conv2D(16, (4, 1),  activation="relu")(x)
    x = Conv2D(32, (4, 1),  activation="relu", strides=4)(x)
    x = Conv2D(64, (4, 1),  activation="relu", strides=4)(x)
    x = Conv2D(128, (4, 1), activation="relu", strides=4)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(2, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model
