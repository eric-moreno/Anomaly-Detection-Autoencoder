import os
import argparse
import numpy as np
import joblib
import h5py as h5
from sklearn.preprocessing import MinMaxScaler
from gwpy.timeseries import TimeSeries


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def augmentation(X_train, timesteps):
    """ Data augmentation process used to extend dataset """
    x = []
    for sample in X_train:
        if sample.shape[0] % timesteps != 0:
            corrected_sample = sample[:-1 * int(sample.shape[0] % timesteps)]
        sliding_sample = np.array([corrected_sample[i:i + timesteps][:] for i in [int(timesteps / 2) * n for n in range(
            int(len(corrected_sample) / (timesteps / 2)) - 1)]])
        x.append(sliding_sample)
    return np.array(x)

def main(args):
    outdir = args.outdir
    detector = args.detector
    freq = args.freq
    filtered = args.filtered
    #eventwidth = args.eventwidth
    filename = args.filename
    os.system('mkdir -p %s'%outdir)
    
    hf = h5.File('%s/%s.h5'%(outdir, filename), 'w')
    
    load_array = ['default_BNS_8sec_14seed.hdf']
    
    for dataset in load_array: 
        
        # Load train 
        load = h5.File('../ggwd/output/'+dataset, 'r')
        
        noise_samples = load['noise_samples']['%s_strain'%(str(detector).lower())][:]
        injection_samples = load['injection_samples']['%s_strain'%(str(detector).lower())][:]
        del load

        # Definining frequency in Hz instead of KHz
        if int(freq) == 2: 
            freq = 2048
        elif int(freq) == 4: 
            freq = 4096

        # With LIGO simulated data, the sample isn't pre-filtered so need to filter again. Real data
        # is not filtered yet. 

        if bool(int(filtered)):
            print('Filtering data with whitening and bandpass')
            print('Sample Frequency: %s Hz'%(freq))
            #randomly distributes GW between (0.2, 0.8) seconds into the event
            #x = [filters(sample, freq)[index:index+2048] for sample, index in zip(data, np.random.randint(9625,10854, size=len(data)))]
            x_noise = [filters(sample, freq) for sample in noise_samples]
            print('Done!')

        if bool(int(filtered)):
            print('Filtering data with whitening and bandpass')
            print('Sample Frequency: %s Hz'%(freq))
            #randomly distributes GW between (0.2, 0.8) seconds into the event
            #x = [filters(sample, freq)[index:index+2048] for sample, index in zip(data, np.random.randint(9625,10854, size=len(data)))]
            x_injection = [filters(sample, freq) for sample in injection_samples]
            print('Done!')
        
        del noise_samples, injection_samples
        
        if dataset == load_array[0]:
            # Normalize the data
            scaler_filename = "%s/scaler_data_unsupervised_noise_L1"%(outdir)
            scaler = joblib.load(scaler_filename) 
            X_noise_transformed = scaler.transform(x_noise)
            X_injection_transformed = scaler.transform(x_injection)
            
            
            #scaler = MinMaxScaler()
            #X_noise_transformed = scaler.fit_transform(x_noise)
            #X_injection_transformed = scaler.transform(x_injection)
            #scaler_filename = "%s/scaler_data_unsupervised_simulated_%s"%(outdir, detector)
            #joblib.dump(scaler, scaler_filename)
        else: 
            X_noise_transformed = scaler.transform(x_noise)
            X_injection_transformed = scaler.transform(x_injection)
  
        if dataset == load_array[0]:                 
            hf.create_dataset('noise', data=X_noise_transformed, maxshape=(None,None))
            hf.create_dataset('injection', data=X_injection_transformed, maxshape=(None,None))
                       
        else: 
            hf["noise"].resize((hf["noise"].shape[0] + X_noise_transformed.shape[0]), axis = 0)
            hf["noise"][-X_noise_transformed.shape[0]:] = X_noise_transformed
            hf["injection"].resize((hf["injection"].shape[0] + X_injection_transformed.shape[0]), axis = 0)
            hf["injection"][-X_injection_transformed.shape[0]:] = X_injection_transformed
            
        del X_noise_transformed, X_injection_transformed, x_noise, x_injection
    hf.close()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required positional arguments
    parser.add_argument("outdir", help="Required output directory")
    parser.add_argument("detector", help="LIGO Detector")
    parser.add_argument("filename", help="Name of processed file")
    # Additional arguments
    parser.add_argument("--freq", help="Sampling frequency of detector in KHz",
                        action='store', dest='freq', default=2)
    parser.add_argument("--filtered", help="Apply LIGO's bandpass and whitening filters",
                        action='store', dest='filtered', default=1)
    parser.add_argument("--eventwidth", help="After processing, how long event a single event should be in seconds",
                        action='store', dest='timesteps', default=1)

    args = parser.parse_args()
    main(args)
