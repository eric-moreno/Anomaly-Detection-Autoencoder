"""
Plot the results produced by the generate_sample.py script.
"""

from __future__ import print_function

import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import h5py as h5


def make_plot():

    plot_path = 'sample.jpg'

    # Load train and test data
    detector = 'L1'
    load = h5.File('../../dataset/default_simulated.hdf', 'r')
    datapoints = len(load['injection_samples']['%s_strain' % (str(detector).lower())])
    noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:datapoints]
    injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:datapoints]
    print("Noise samples shape:", noise_samples.shape)
    print("Injection samples shape:", injection_samples.shape)

    features = np.concatenate((noise_samples, injection_samples))

    gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
    noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
    targets = np.transpose(np.array([gw, noise]))

    # splitting the train / test data in ratio 80:20
    train_data, test_data, train_truth, test_truth = train_test_split(features, targets, test_size=0.2)

    train_data = train_data.reshape((train_data.shape[0], 1, -1))
    print("Train data shape:", train_data.shape)
    train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
    print("Train labels data shape:", train_truth.shape)
    test_data = test_data.reshape((test_data.shape[0], 1, -1))
    print("Test data shape:", test_data.shape)
    test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
    print("Test labels data shape:", test_truth.shape)

    n_plots = 1
    for i in range(n_plots):
        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        print(test_data[i][0])
        plt.plot(test_data[i][0], color='C0')
        plt.subplot(2, 1, 2)
        print(test_truth[i][0][0])
        plt.plot([0], [test_truth[i][0][0]], marker='o', markersize=3, color="red", label='Labels: 0 - noise, 1 - gw')
        plt.legend(loc='upper left')

        plt.suptitle(f'Sample {i}. Train data (top) and label (bottom)')

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    make_plot()
