""" Check data format and shapes for SNN Toolbox """

import numpy as np


def check_data(train, test):
    with np.load(train) as train_data:
        print("\n Train Data:")
        print("Keys: ", train_data.files)
        for file in train_data.files:
            train_shape = train_data[file].shape
            print("Dimensions: ", train_shape)

    with np.load(test) as test_data:
        print("\n Test Data:")
        print("Keys: ", test_data.files)
        for file in test_data.files:
            test_shape = test_data[file].shape
            print("Dimensions: ", test_shape)


if __name__ == "__main__":
    train_path = 'x_test.npz'
    test_path = 'y_test.npz'
    check_data(train_path, test_path)
