import os
import argparse
import numpy as np
import joblib
import time 
import seaborn as sns
import matplotlib.pyplot as plt
import h5py as h5
import setGPU
from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from gwpy.timeseries import TimeSeries
import torch.nn as nn
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector, Conv1D, \
    MaxPooling1D, UpSampling1D, Flatten, Reshape, GRU
from keras.models import Model
from keras import regularizers
from multiprocessing import Process

def augmentation(X_train, timesteps, divide_into):
    """ Data augmentation process used to extend dataset """
    x = []
    for sample in X_train:
        if sample.shape[0] % timesteps != 0:
            sample = sample[int(sample.shape[0] % timesteps):]
        
        sliding_sample = np.array([sample[i:i + timesteps] for i in [int(timesteps / divide_into) * n for n in range(
                                    int(len(sample) / (timesteps / divide_into)) - (divide_into - 1))]])
        x.append(sliding_sample)
    return np.array(x)


def main(args, detector, sample_type): 
    
    outdir = args.outdir
    #detector = args.detector
    #sample_type = args.datatype
    scaler_flag = bool(args.scaler)
    timesteps = int(args.timesteps)
    model_arch = str(args.model)
    os.system('mkdir -p %s' % outdir)
    num_steps = int(2048*5.5)
    
    clip = False
    badremoved = True

    if detector == 'Both': 
        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed3.hdf', 'r')
        X_train = load[sample_type + '_samples']['L1'.lower() + '_strain'][:, :num_steps]
        X_train_temp = load[sample_type + '_samples']['H1'.lower() + '_strain'][:, :num_steps]
        X_train = np.concatenate((X_train.reshape(-1, X_train.shape[1], 1), 
                                  X_train_temp.reshape(-1, X_train.shape[1], 1)), axis=2)
        
        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed2.hdf', 'r')
        X_train_temp_1 = load[sample_type + '_samples']['L1'.lower() + '_strain'][:, :num_steps]
        X_train_temp_2 = load[sample_type + '_samples']['H1'.lower() + '_strain'][:, :num_steps]
        X_train_temp = np.concatenate((X_train_temp_1.reshape(-1, X_train.shape[1], 1), 
                                       X_train_temp_2.reshape(-1, X_train.shape[1], 1)), axis=2)
        X_train = np.concatenate((X_train, X_train_temp))

        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed1.hdf', 'r')
        X_train_temp_1 = load[sample_type + '_samples']['L1'.lower() + '_strain'][:, :num_steps]
        X_train_temp_2 = load[sample_type + '_samples']['H1'.lower() + '_strain'][:, :num_steps]
        X_train_temp = np.concatenate((X_train_temp_1.reshape(-1, X_train.shape[1], 1), 
                                       X_train_temp_2.reshape(-1, X_train.shape[1], 1)), axis=2)
        X_train = np.concatenate((X_train, X_train_temp))      
        
    else: 
        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed3.hdf', 'r')
        X_train = load[sample_type + '_samples'][detector.lower() + '_strain'][:, :num_steps]
        #X_train_temp = load['injection_samples'][detector.lower() + '_strain'][:, :num_steps]
        #X_train = np.concatenate((X_train, X_train_temp))

        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed2.hdf', 'r')
        X_train_temp = load[sample_type + '_samples'][detector.lower() + '_strain'][:, :num_steps]
        X_train = np.concatenate((X_train, X_train_temp))
        #X_train_temp = load['injection_samples'][detector.lower() + '_strain'][:, :num_steps]
        #X_train = np.concatenate((X_train, X_train_temp))

        load = h5.File('../ggwd/output/updated_BBH_8sec_SEOBNRv4_realdata_seed1.hdf', 'r')
        X_train_temp = load[sample_type + '_samples'][detector.lower() + '_strain'][:, :num_steps]
        X_train = np.concatenate((X_train, X_train_temp))
        #X_train_temp = load['injection_samples'][detector.lower() + '_strain'][:, :num_steps]
        #X_train = np.concatenate((X_train, X_train_temp))

    np.random.shuffle(X_train)

    if clip: 
        X_train = np.clip(X_train, -150, 150)
        scaler = joblib.load('standard_scaler_' + detector.lower() + '_clip')

    elif badremoved: 
        X_train_clean = []
        for i in range(len(X_train)): 
            if X_train[i].max() < 150: 
                X_train_clean.append(X_train[i])
        X_train = np.array(X_train_clean)
        del X_train_clean
        scaler = joblib.load('standard_scaler_' + detector.lower() + '_realdata_noise')

    len_X_train = len(X_train)

    # Create Scaler     
    #scaler = StandardScaler()
    #X_train = scaler.fit_transform(X_train.reshape((-1, 1)))
    #X_train = X_train.reshape((len_X_train, -1))
    #joblib.dump(scaler, 'standard_scaler_' + detector + '_realdata_noise')
    
    
    # Scaler already produced 
    if scaler_flag:
        print('SCALING')
        X_train = scaler.transform(X_train.reshape((-1, 1))).reshape((len_X_train, -1))

    print(X_train.shape)
    #X_train = augmentation(X_train, timesteps, 2)
    
    if detector == 'Both': 
        X_train = X_train.reshape((-1, timesteps, 2))
                                 
    else:
        X_train = X_train.reshape((-1, timesteps, 1))
    print(X_train.shape)
    
    def autoencoder_LSTM(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(48, activation='tanh', return_sequences=True, 
                  kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(24, activation='tanh', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(24, activation='tanh', return_sequences=True)(L3)
        L5 = LSTM(48, activation='tanh', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)    
        model = Model(inputs=inputs, outputs=output)
        return model

    def autoencoder_Conv_paper(X): 
        inputs = Input(shape=(X.shape[1],X.shape[2]))
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

    def autoencoder_GRU(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = GRU(48, activation='tanh', return_sequences=True, 
                  kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = GRU(24, activation='tanh', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = GRU(24, activation='tanh', return_sequences=True)(L3)
        L5 = GRU(48, activation='tanh', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)    
        model = Model(inputs=inputs, outputs=output)
        return model

    epochs = 100
    batch_size = 16
    
    if model_arch == 'LSTM': 
        model = autoencoder_LSTM(X_train)
    elif model_arch == 'GRU': 
        model = autoencoder_GRU(X_train)
    elif model_arch == 'CNN': 
        model = autoencoder_Conv_paper(X_train)
    
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=7, verbose=0, mode='min')
      
    if model_arch == 'LSTM': 
        mcp_save = ModelCheckpoint(outdir + '/best_badremoved_standard_LSTM_' + detector.lower() + '_'+ sample_type + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    elif model_arch == 'GRU': 
        mcp_save = ModelCheckpoint(outdir + '/best_badremoved_standard_GRU_' + detector.lower() + '_'+ sample_type + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    elif model_arch == 'CNN': 
        mcp_save = ModelCheckpoint(outdir + '/best_badremoved_standard_CNN_' + detector.lower() + '_'+ sample_type + '.hdf5', save_best_only=True, monitor='val_loss', mode='min')
    

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[early_stop, mcp_save])

    del X_train 
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    #parser.add_argument("detector", help="LIGO Detector")
    #parser.add_argument("datatype", help="noise or injection")
    parser.add_argument("timesteps", help="How many timesteps fed to AE models")
    parser.add_argument("model", help="LSTM, GRU, or CNN")
    parser.add_argument("--scaler", action='store_true', help="Turn on and off scaler (produced in preprocess)")

    args_parser = parser.parse_args()
    #for detector in ['H1', 'L1']:
    for detector in ['Both']:
        for datatype in ['noise', 'injection']:
            print('Training following parameters: ')
            print('Detector: %s'%(detector))
            print('Datatype: %s'%(datatype))
            main(args_parser, detector, datatype) 
    '''
    for detector in ['H1', 'L1']: 
        for datatype in ['noise', 'injection']:
            process = Process(target=main, args = (args_parser, detector, datatype))
            process.start()
            process.join()
            
    '''
    
