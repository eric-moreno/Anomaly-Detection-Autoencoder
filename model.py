from numpy.random import seed
from tensorflow import set_random_seed
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR) 

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Flatten, Reshape
from keras.models import Model
from keras import regularizers, Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse

def autoencoder_LSTM(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(32, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(8, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(8, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(32, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_Conv(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(encoded) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(16, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model 

def autoencoder_ConvDNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(32, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(2, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    x = Flatten()(encoded)
    x = Dense(64, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(20, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(130, activation='relu')(x)
    x = Reshape((13, 10))(x)
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(x) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(32, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(2)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model 

def autoencoder_ConvLSTM(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = Conv1D(16, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(4, padding="same")(L1) # 5 dims
    L3 = Conv1D(10, 3, activation="relu", padding="same")(L2) # 5 dims
    #x = BatchNormalization()(x)
    encoded = MaxPooling1D(4, padding="same")(L3) # 3 dims
    x = Reshape((70, 1))(encoded)
    
    x = LSTM(32, activation='relu', return_sequences=False, 
              kernel_regularizer=regularizers.l2(0.00))(x)
    x = RepeatVector(70)(x)
    x = LSTM(32, activation='relu', return_sequences=True)(x)
    out = TimeDistributed(Dense(1))(x)  
    
    x = Reshape((7, 10))(out)
    # 3 dimensions in the encoded layer
    L4 = Conv1D(10, 3, activation="relu", padding="same")(x) # 3 dims
    #x = BatchNormalization()(x)
    L5 = UpSampling1D(4)(L4) # 6 dims
    L6 = Conv1D(32, 2, activation='relu')(L5) # 5 dims
    #x = BatchNormalization()(x)
    L7 = UpSampling1D(4)(L6) # 10 dims
    output = Conv1D(1, 3, activation='sigmoid', padding='same')(L7) # 10 dims
    model = Model(inputs=inputs, outputs = output)
    return model 

def autoencoder_DeepConv(X):
    
    ### Use autoencoder_ConvDNN instead ###
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Conv1D(16, 16, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(32, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Conv1D(64, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = MaxPooling1D(4, padding="same")(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(64, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(32, 8, activation="relu", padding="same",dilation_rate=4)(x)
    x = UpSampling1D(4)(x)
    x = Conv1D(16, 16, activation="relu", padding="same")(inputs)
    x = Dense(X.shape[1], activation='relu')(x)
    output = Reshape((X.shape[1], 1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    x = Flatten()(inputs)
    x = Dense(int(X.shape[1]/2), activation='relu')(x)
    x = Dense(int(X.shape[1]/10), activation='relu')(x)
    x = Dense(int(X.shape[1]/2), activation='relu')(x)
    x = Dense(X.shape[1], activation='relu')(x)
    output = Reshape((X.shape[1], 1))(x)
    model = Model(inputs=inputs, outputs=output)
    return model
