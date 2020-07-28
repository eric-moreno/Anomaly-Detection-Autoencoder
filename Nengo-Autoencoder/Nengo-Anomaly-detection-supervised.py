import os
import collections
import warnings

import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import h5py as h5
from sklearn.preprocessing import MinMaxScaler
from gwpy.timeseries import TimeSeries
import joblib
from sklearn.utils import shuffle

from keras.layers import Input, Dense, Flatten, Reshape
from keras.models import Model

import nengo_loihi


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


# ignore NengoDL warning about no GPU
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# The results in this notebook should be reproducible across many random seeds.
# However, some seed values may cause problems, particularly in the `to-spikes` layer
# where poor initialization can result in no information being sent to the chip. We set
# the seed to ensure that good results are reproducible without having to re-train.
np.random.seed(0)
tf.random.set_seed(0)

outdir = "Outputs"
detector = "L1"
freq = 2
filtered = 1
timesteps = 100
os.system(f'mkdir {outdir}')

# Load train and test data
load = h5.File('../../dataset/default_simulated.hdf', 'r')

# Define frequency in Hz instead of KHz
if int(freq) == 2:
    freq = 2048
elif int(freq) == 4:
    freq = 4096
else:
    print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

datapoints = 5000
noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:datapoints]
injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:datapoints]
train_data = np.concatenate((noise_samples, injection_samples))
train_truth = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
train_data, train_truth = shuffle(train_data, train_truth)

print("Noise samples shape:", noise_samples.shape)
print("Injection samples shape:", injection_samples.shape)

# With LIGO simulated data, the sample isn't pre-filtered so need to filter again. Real data
# is not filtered yet.
# if bool(int(filtered)):
#     print('Filtering data with whitening and bandpass')
#     print('Sample Frequency: %s Hz' % (freq))
#     x = [filters(sample, freq)[7168:15360] for sample in train_data]
#     print('Done!')

# # Normalize the data
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(noise_samples)
# scaler_filename = f"{outdir}/scaler_data_{detector}"
# joblib.dump(scaler, scaler_filename)


X_train = train_data.reshape((train_data.shape[0], 1, -1))
train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))

# # Reshape inputs
# X_train = X_train.reshape(-1, timesteps, 1)
print("Training data shape:", X_train.shape)
# np.savez('x_test.npz', arr_0=X_train)
# train_truth = train_truth.reshape(-1, timesteps, 1)
print("Test data shape:", train_truth.shape)
# np.savez('y_test.npz', arr_0=X_train)
# print("Test and Train data saved in npz format")

# train_truth = X_train

#
# inputs = Input(shape=(X_train.shape[1], X_train.shape[2]), name="input")
# x = Flatten()(inputs)
#
# # transform input signal to spikes using trainable off-chip layer
# to_spikes_layer = Dense(X_train.shape[2], activation='relu')(x)
#
# # on-chip dense layers
# x = Dense(int(X_train.shape[2] / 2), activation='relu')(to_spikes_layer)
# x = Dense(int(X_train.shape[2] / 10), activation='relu')(x)
# x = Dense(int(X_train.shape[2] / 2), activation='relu')(x)
# x = Dense(X_train.shape[2], activation='relu')(x)
#
# # since this final output layer has no activation function,
# # it will be converted to a `nengo.Node` and run off-chip
# output = Reshape((1, X_train.shape[2]))(x)
# model = Model(inputs=inputs, outputs=output)
# model.summary()

# # Reshape inputs for LSTM
# # X_train = X_train.reshape(-1, 1, timesteps)
# print("Training data shape:", X_train.shape)
# np.savez('x_test.npz', arr_0=X_train)
# # X_test = X_test.reshape(-1, 1, timesteps)
# print("Test data shape:", train_truth.shape)
# np.savez('y_test.npz', arr_0=train_truth)
# print("Test and Train data saved in npz format")

# inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))
# x = Flatten()(inputs)
# x = Dense(int(X_train.shape[2] / 2), activation='relu')(x)
# x = Dense(int(X_train.shape[2] / 10), activation='relu')(x)
# x = Dense(int(X_train.shape[2] / 2), activation='relu')(x)
# x = Dense(X_train.shape[2], activation='relu')(x)
# output = Reshape((X_train.shape[2], X_train.shape[1]))(x)
# model = Model(inputs=inputs, outputs=output)

inp = Input(shape=(X_train.shape[2],), name="input")

L1_layer = Dense(1024, activation='relu')
L1 = L1_layer(inp)

L2_layer = Dense(128, activation='relu')
L2 = L2_layer(L1)

L3_layer = Dense(64, activation='relu')
L3 = L3_layer(L2)

output = Dense(1, activation='relu')(L3)

model = Model(inputs=inp, outputs=output)
# model.compile(optimizer='adam', loss='mse')
model.summary()


# fit the model to the data
# nb_epochs = 300
# batch_size = 16
# history = model.fit(X_train, train_truth, epochs=nb_epochs, batch_size=batch_size,
#                     validation_split=0.2,).history


def train(params_file="./keras_to_loihi_params", epochs=1, **kwargs):
    converter = nengo_dl.Converter(model, **kwargs)

    with nengo_dl.Simulator(converter.net, seed=0, minibatch_size=100) as sim:
        sim.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={converter.outputs[output]: tf.keras.losses.MeanSquaredError()},
            metrics={converter.outputs[output]: tf.keras.metrics.MeanSquaredError()},
        )
        sim.fit(
            {converter.inputs[inp]: X_train},
            {converter.outputs[output]: train_truth},
            epochs=epochs,
        )

        # save the parameters to file
        sim.save_params(params_file)


# train this network with normal ReLU neurons
train(epochs=2, swap_activations={tf.nn.relu: nengo.RectifiedLinear()})

# just to compile the model for now
test_images = train_data
test_labels = train_truth


def run_network(
        activation,
        params_file="./keras_to_loihi_params",
        n_steps=30,
        scale_firing_rates=1,
        synapse=None,
        n_test=100,
        n_plots=1,
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        scale_firing_rates=scale_firing_rates,
        swap_activations={tf.nn.relu: activation},
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[output]

    # add probes to layers to record activity
    with nengo_converter.net:
        probes = collections.OrderedDict([[L1_layer, nengo.Probe(nengo_converter.layers[L1])],
                                          [L2_layer, nengo.Probe(nengo_converter.layers[L2])],
                                          [L3_layer, nengo.Probe(nengo_converter.layers[L3])],])

    # repeat inputs for some number of timesteps
    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=1, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})

    # compute accuracy on test data, using output of network on
    # last timestep
    test_predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    print(
        "Test accuracy: %.2f%%"
        % (100 * np.mean(test_predictions == test_labels[:n_test, 0, 0]))
    )

    # plot the results
    mean_rates = []
    for i in range(n_plots):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        # plt.title("Input image")
        # plt.imshow(test_images[i, 0].reshape((28, 28)), cmap="gray")
        # plt.axis("off")

        n_layers = len(probes)
        mean_rates_i = []
        for j, layer in enumerate(probes.keys()):
            probe = probes[layer]
            plt.subplot(n_layers, 3, (j * 3) + 2)
            plt.suptitle("Neural activities")

            outputs = data[probe][i]

            # look at only at non-zero outputs
            nonzero = (outputs > 0).any(axis=0)
            outputs = outputs[:, nonzero] if sum(nonzero) > 0 else outputs

            # undo neuron amplitude to get real firing rates
            outputs /= nengo_converter.layers[layer].ensemble.neuron_type.amplitude

            rates = outputs.mean(axis=0)
            mean_rate = rates.mean()
            mean_rates_i.append(mean_rate)
            print(
                '"%s" mean firing rate (example %d): %0.1f' % (layer.name, i, mean_rate)
            )

            if is_spiking_type(activation):
                outputs *= 0.001
                plt.ylabel("# of Spikes")
            else:
                plt.ylabel("Firing rates (Hz)")

            # plot outputs of first 100 neurons
            plt.plot(outputs[:, :100])

        mean_rates.append(mean_rates_i)

        plt.xlabel("Timestep")

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(tf.nn.softmax(data[nengo_output][i]))
        plt.legend([str(j) for j in range(10)], loc="upper left")
        plt.xlabel("Timestep")
        plt.ylabel("Probability")

        plt.tight_layout()

    # take mean rates across all plotted examples
    mean_rates = np.array(mean_rates).mean(axis=0)

    return mean_rates


def is_spiking_type(neuron_type):
    return isinstance(neuron_type, (nengo.LIF, nengo.SpikingRectifiedLinear))

# test the trained networks on test set
mean_rates = run_network(activation=nengo.RectifiedLinear(), n_steps=10)

# test the trained networks using spiking neurons
run_network(
    activation=nengo.SpikingRectifiedLinear(), scale_firing_rates=100, synapse=0.005,
)

# test the trained networks using spiking neurons
run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=100,
    synapse=0.005,
)


def plot_activation(neurons, min, max, **kwargs):
    x = np.arange(min, max, 0.001)
    fr = neurons.rates(x=x, gain=[1], bias=[0])

    plt.plot(x, fr, lw=2, **kwargs)
    plt.title("%s with [gain=1, bias=0]" % str(neurons))
    plt.ylabel("Firing rate (Hz)")
    plt.xlabel("Input signal")
    plt.legend(["Standard", "Loihi"], loc=2)


plt.figure(figsize=(10, 3))
plot_activation(nengo.RectifiedLinear(), -100, 1000)
plot_activation(nengo_loihi.neurons.LoihiSpikingRectifiedLinear(), -100, 1000)

plt.figure(figsize=(10, 3))
plot_activation(nengo.LIF(), -4, 40)
plot_activation(nengo_loihi.neurons.LoihiLIF(), -4, 40)

target_mean = 200
scale_firing_rates = {
    L1_layer: target_mean / mean_rates[0],
    L2_layer: target_mean / mean_rates[1],
    L3_layer: target_mean / mean_rates[2],
}

# test the trained networks using spiking neurons
run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=scale_firing_rates,
    synapse=0.005,
)

# train this network with normal ReLU neurons
train(
    params_file="./keras_to_loihi_loihineuron_params",
    epochs=2,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    scale_firing_rates=100,
)

# test the trained networks using spiking neurons
run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=100,
    params_file="./keras_to_loihi_loihineuron_params",
    synapse=0.005,
)

pres_time = 0.03  # how long to present each input, in seconds
n_test = 5  # how many images to test

# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    model,
    scale_firing_rates=400,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    synapse=0.005,
)
net = nengo_converter.net

# get input/output objects
nengo_input = nengo_converter.inputs[inp]
nengo_output = nengo_converter.outputs[output]

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

# build network, load in trained weights, save to network
with nengo_dl.Simulator(net) as nengo_sim:
    nengo_sim.load_params("keras_to_loihi_loihineuron_params")
    nengo_sim.freeze_params(net)

with net:
    nengo_input.output = nengo.processes.PresentInput(
        test_images, presentation_time=pres_time
    )

with net:
    nengo_loihi.add_params(net)  # allow on_chip to be set
    net.config[nengo_converter.layers[L1].ensemble].on_chip = False

with net:
    L2_shape = L2_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[L2].ensemble
    ].block_shape = nengo_loihi.BlockShape((50,), L2_shape)

    L3_shape = L3_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[L3].ensemble
    ].block_shape = nengo_loihi.BlockShape((50,), L3_shape)


# build Nengo Loihi Simulator and run network
with nengo_loihi.Simulator(net) as loihi_sim:
    loihi_sim.run(n_test * pres_time)

    # get output (last timestep of each presentation period)
    pres_steps = int(round(pres_time / loihi_sim.dt))
    output = loihi_sim.data[nengo_output][pres_steps - 1 :: pres_steps]

    # compute the Loihi accuracy
    loihi_predictions = np.argmax(output, axis=-1)
    correct = 100 * np.mean(loihi_predictions == test_labels[:n_test, 0, 0])
    print("Loihi accuracy: %.2f%%" % correct)

# plot the neural activity of the convnet layers
plt.figure(figsize=(12, 4))

timesteps = loihi_sim.trange() / loihi_sim.dt

# plot the presented MNIST digits
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
images = test_images.reshape(-1, 28, 28, 1)[:n_test]
ni, nj, nc = images[0].shape
allimage = np.zeros((ni, nj * n_test, nc), dtype=images.dtype)
for i, image in enumerate(images[:n_test]):
    allimage[:, i * nj : (i + 1) * nj] = image
if allimage.shape[-1] == 1:
    allimage = allimage[:, :, 0]
plt.imshow(allimage, aspect="auto", interpolation="none", cmap="gray")
plt.xticks([])
plt.yticks([])

# plot the network predictions
plt.subplot(2, 1, 2)
plt.plot(timesteps, loihi_sim.data[nengo_output])
plt.legend(["%d" % i for i in range(10)], loc="lower left")
plt.suptitle("Output predictions")
plt.xlabel("Timestep")
plt.ylabel("Probability")
