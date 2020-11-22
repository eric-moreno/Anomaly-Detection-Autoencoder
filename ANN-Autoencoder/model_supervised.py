from numpy.random import seed
import tensorflow as tf
#tf.logging.set_verbosity(tf.logging.ERROR) 

from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Flatten, Reshape, GRU, Conv2D
from keras.models import Model
from keras import regularizers, Sequential 

def autoencoder_ConvDNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Reshape((X.shape[1], 1))(inputs)
    L1 = Conv1D(16, 4, activation="relu", dilation_rate=1)(x)
    L2 = MaxPooling1D(2, stride=4)(L1)
    L3 = Conv1D(32, 4, activation="relu", dilation_rate=2)(L2)
    L4 = MaxPooling1D(2, stride=4)(L3) 
    L5 = Conv1D(64, 4, activation="relu", dilation_rate=2)(L4)
    L6 = MaxPooling1D(4, stride=2)(L5)
    L7 = Conv1D(128, 8, activation="relu", dilation_rate=2)(L6)
    L8 = MaxPooling1D(4, stride=2)(L7)
    x = Flatten()(L8)
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs = output)
    return model  

def autoencoder_ConvDNN_Nengo(X):
    # replicating exact architecture used in Nengo
    
    inp = Input(shape=(1, X.shape[2]), name="input")
    x = Reshape((X.shape[2], 1, 1))(inp)
    
    to_spikes_layer = Conv2D(16, (4, 1), activation=tf.nn.relu, use_bias=False)
    to_spikes = to_spikes_layer(x)
    
    L1_layer = Conv2D(16, (4, 1), strides=4, activation=tf.nn.relu, use_bias=False)
    L1 = L1_layer(to_spikes)

    L2_layer = Conv2D(32, (4, 1), strides=4, activation=tf.nn.relu, use_bias=False)
    L2 = L2_layer(L1)

    L3_layer = Conv2D(64, (4, 1), strides=4, activation=tf.nn.relu, use_bias=False)
    L3 = L3_layer(L2)

    L4_layer = Conv2D(128, (8, 1), strides=4, activation=tf.nn.relu, use_bias=False)
    L4 = L4_layer(L3)

    x = Flatten()(L4)

    L5_layer = Dense(128, activation=tf.nn.relu, use_bias=False)
    L5 = L5_layer(x)

    L6_layer = Dense(64, activation=tf.nn.relu, use_bias=False)
    L6 = L6_layer(L5)

    # since this final output layer has no activation function,
    # it will be converted to a `nengo.Node` and run off-chip
    output = Dense(units=2, name="output")(L6)

    model = Model(inputs=inp, outputs=output)
    return model

def autoencoder_DNN(X):
    inputs = Input(shape=(X.shape[1],))
    x = Dense(1024, activation='relu')(inputs)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, activation='relu')(x)
    model = Model(inputs=inputs, outputs=output)
    return model

