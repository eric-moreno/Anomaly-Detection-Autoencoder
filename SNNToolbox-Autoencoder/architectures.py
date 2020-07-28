""" Autoencoder models convertable to SNNs for anomaly detection. """

from keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape
from keras.models import Model


def autoencoder_Conv(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs)  # 10 dims
    # x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1)  # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2)  # 5 dims
    # x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3)  # 3 dims
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(encoded)  # 3 dims
    # x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4)  # 6 dims
    L6 = Conv1D(16, 2, activation='relu')(L5)  # 5 dims
    # x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6)  # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7)  # 10 dims
    model = Model(inputs=inputs, outputs=output)
    return model


def autoencoder_ConvDNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(32, 3, activation="relu", padding="same")(inputs)  # 10 dims
    # x = BatchNormalization()(x)
    L2 = MaxPooling1D(2, padding="same")(L1)  # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2)  # 5 dims
    # x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3)  # 3 dims
    x = Flatten()(encoded)
    x = Dense(64, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(130, activation='relu')(x)
    x = Reshape((13, 10))(x)
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(x)  # 3 dims
    # x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4)  # 6 dims
    L6 = Conv1D(32, 2, activation='relu')(L5)  # 5 dims
    # x = BatchNormalization()(x)
    L7 = UpSampling1D(2)(L6)  # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7)  # 10 dims
    model = Model(inputs=inputs, outputs=output)
    return model


def autoencoder_DeepConv(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(16, 16, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(32, 8, activation="relu", padding="same", dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(64, 8, activation="relu", padding="same", dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(64, 8, activation="relu", padding="same", dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(32, 8, activation="relu", padding="same", dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(16, 16, activation="relu", padding="same")(x)
    x = Dense(X.shape[1], activation='relu')(x)
    output = Reshape((X.shape[1], 1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1]))
    x = Dense(int(X.shape[1] / 2), activation='relu')(inputs)
    x = Dense(int(X.shape[1] / 10), activation='relu')(x)
    x = Dense(int(X.shape[1] / 2), activation='relu')(x)
    output = Dense(X.shape[1], activation='relu')(x)
    model = Model(inputs=inputs, outputs=output)
    return model
