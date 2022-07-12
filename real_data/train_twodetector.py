import os
import argparse
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
import setGPU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from gwpy.timeseries import TimeSeries
import torch.nn as nn
import neuroaikit as ai
import neuroaikit.tf as aitf
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, \
    MaxPooling1D, UpSampling1D, Flatten, Reshape, GRU
from keras.models import Model
from keras import regularizers

timesteps = 100
num_steps = 16300

clip = False
badremoved = True

load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_noise.hdf', 'r')
X_train_l1 = load['noise_samples']['l1_strain'][:, :num_steps]
X_train_h1 = load['noise_samples']['h1_strain'][:, :num_steps]

detector = 'l1'
if clip: 
    X_train_l1 = np.clip(X_train_h1, -150, 150)
    scaler = joblib.load('standard_scaler_' + detector.lower() + '_clip')
    
elif badremoved: 
    X_train_clean = []
    for i in range(len(X_train_l1)): 
        if X_train_l1[i].max() < 150: 
            X_train_clean.append(X_train_l1[i])
    X_train_l1 = np.array(X_train_clean)
    scaler = joblib.load('standard_scaler_' + detector.lower() + '_badremoved')

X_train_l1 = scaler.transform(X_train_l1.reshape((-1, 1))).reshape((-1, timesteps, 1))
    
detector = 'h1'
if clip: 
    X_train = np.clip(X_train, -150, 150)
    scaler = joblib.load('standard_scaler_' + detector.lower() + '_clip')
    
elif badremoved: 
    X_train_clean = []
    for i in range(len(X_train_h1)): 
        if X_train_h1[i].max() < 150: 
            X_train_clean.append(X_train_h1[i])
    X_train_h1 = np.array(X_train_clean)
    scaler = joblib.load('standard_scaler_' + detector.lower() + '_badremoved')

X_train_h1 = scaler.transform(X_train_h1.reshape((-1, 1))).reshape((-1, timesteps, 1))



#X_train = np.concatenate((X_train_l1, X_train_h1))
np.random.shuffle(X_train)
print(X_train.shape)

def autoencoder_LSTM(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(32, activation='tanh', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(12, activation='tanh', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(12, activation='tanh', return_sequences=True)(L3)
    L5 = LSTM(32, activation='tanh', return_sequences=True)(L4)
    output = TimeDistributed(Dense(X.shape[2]))(L5)    
    model = Model(inputs=inputs, outputs=output)
    return model

def autoencoder_Conv_paper(X): 
    inputs = Input(shape=(X.shape[1],1))
    L1 = Conv1D(256, 3, activation="relu", padding="same")(inputs) # 10 dims
    #x = BatchNormalization()(x)
    L2 = MaxPooling1D(2, padding="same")(L1) # 5 dims
    encoded = Conv1D(128, 3, activation="relu", padding="same")(L2) # 5 dims
    # 3 dimensions in the encoded layer
    L3 = UpSampling1D(2)(encoded) # 6 dims
    L4 = Conv1D(256, 3, activation='relu', padding="same")(L3)
    output = Conv1D(1, 3, activation='sigmoid', padding="same")(L4)
    model = Model(inputs=inputs, outputs = output)
    return model 

epochs = 100
batch_size = 2048
model = autoencoder_LSTM(X_train)
model.summary()
early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')

if clip: 
    mcp_save = ModelCheckpoint('trained_models/best_clip_standard_' + 'l1h1' + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')

elif badremoved:     
    mcp_save = ModelCheckpoint('trained_models/best_badremoved_standard_' + 'l1h1' + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    
model.compile(optimizer='adam', loss='mse', metrics=['mse'])
history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop, mcp_save])