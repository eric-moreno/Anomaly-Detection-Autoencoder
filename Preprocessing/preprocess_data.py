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
    
    # Load train 
    load_1 = h5.File('../../dataset/default_simulated_big_1.hdf', 'r')
    load_2 = h5.File('../../dataset/default_simulated_big_2.hdf', 'r')
    load_3 = h5.File('../../dataset/default_simulated_big_3.hdf', 'r')
    
    datapoints = 40000
    noise_samples_1 = load_1['noise_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    injection_samples_1 = load_1['injection_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    noise_samples_2 = load_2['noise_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    injection_samples_2 = load_2['injection_samples']['%s_strain'%(str(detector).lower())][:datapoints]
    noise_samples_3 = load_3['noise_samples']['%s_strain'%(str(detector).lower())][:int(datapoints)]
    injection_samples_3 = load_3['injection_samples']['%s_strain'%(str(detector).lower())][:int(datapoints)]
    del load_1, load_2, load_3
    data = np.concatenate((injection_samples_1, injection_samples_2, injection_samples_3, noise_samples_1, noise_samples_2 ,noise_samples_3))
    truth = np.concatenate((np.ones(int(datapoints*3)), np.zeros(int(datapoints*3))))

    del noise_samples_1, noise_samples_2 ,noise_samples_3, injection_samples_1, injection_samples_2, injection_samples_3
    
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
        x = [filters(sample, freq)[10240:12288] for sample in data]
        print('Done!')
    # Normalize the data
    scaler = MinMaxScaler()
    X_transformed = scaler.fit_transform(x)
    scaler_filename = "%s/scaler_data_%s"%(outdir, detector)
    joblib.dump(scaler, scaler_filename)
    
    hf = h5.File('%s/%s.h5'%(outdir, filename), 'w')
    hf.create_dataset('data', data=X_transformed)
    hf.create_dataset('truth', data=truth)
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
    parser.add_argument("--eventwidth", help="After processing, how long event a single event should be.",
                        action='store', dest='timesteps', default=1)

    args = parser.parse_args()
    main(args)
