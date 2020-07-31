""" Supervised anomaly detection in LIGO data with Spiking Neural Networks running on Loihi """

import os
import collections
import warnings

import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import h5py as h5
import joblib
from gwpy.timeseries import TimeSeries
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from keras.layers import Input, Dense
from keras.models import Model

import nengo_loihi


def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """ This function prints and plots the confusion matrix. Normalization can be added by setting `normalize=True` """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print(title)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes,
           title=title, ylabel='True label', xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


def print_neurons_type(converter_nengo):
    print("Types of neurons used: ")
    for ensemble in converter_nengo.net.ensembles:
        print(ensemble, ensemble.neuron_type)


# Ignore NengoDL warning about no GPU
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# The results in this training should be reproducible across many random seeds.
# However, some seed values may cause problems, particularly in the `to-spikes` layer
# where poor initialization can result in no information being sent to the chip.
# We set the seed to ensure that good results are reproducible without having to re-train.
# np.random.seed(0)
# tf.random.set_seed(0)

outdir = "Output_DNN"
detector = "L1"
freq = 2
filtered = 1
timesteps = 100
os.system(f'mkdir {outdir}')

# counter of plots to save
plot_no = 0

# Load train and test data
load = h5.File('../../dataset/240k_1sec_L1.h5', 'r')

# Define frequency in Hz instead of KHz
if int(freq) == 2:
    freq = 2048
elif int(freq) == 4:
    freq = 4096
else:
    print(f'Given frequency {freq}kHz is not supported. Correct values are 2 or 4kHz.')

#datapoints = len(load['injection_samples']['%s_strain' % (str(detector).lower())])
#noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:datapoints]
#injection_samples = load['injection_samples']['%s_strain' % (str(detector).lower())][:datapoints]
#print("Noise samples shape:", noise_samples.shape)
#print("Injection samples shape:", injection_samples.shape)

#features = np.concatenate((noise_samples, injection_samples))
# targets = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
datapoints = 120000
gw = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
noise = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
targets = np.transpose(np.array([gw, noise]))

X_train = load['data'][:]
# splitting the train / test data in ratio 80:20
train_data, test_data, train_truth, test_truth = train_test_split(X_train, targets, test_size=0.2)
class_names = np.array(['noise', 'GW'], dtype=str)


# With LIGO simulated data, the sample isn't pre-filtered so need to filter again. Real data is not filtered yet.
#if bool(int(filtered)):
#    print('Filtering data with whitening and bandpass')
#   print('Sample Frequency: %s Hz' % freq)
#    x = [filters(sample, freq)[7168:15360] for sample in train_data]
#    print('Done!')

# Normalize the data
# scaler = MinMaxScaler()
# train_data = scaler.fit_transform(noise_samples)
# scaler_filename = f"{outdir}/scaler_data_{detector}"
# joblib.dump(scaler, scaler_filename)

# Reshape inputs
train_data = train_data.reshape((train_data.shape[0], 1, -1))
print("Train data shape:", train_data.shape)
train_truth = train_truth.reshape((train_truth.shape[0], 1, -1))
print("Train labels data shape:", train_truth.shape)
test_data = test_data.reshape((test_data.shape[0], 1, -1))
print("Test data shape:", test_data.shape)
test_truth = test_truth.reshape((test_truth.shape[0], 1, -1))
print("Test labels data shape:", test_truth.shape)

# Define the model
inp = Input(shape=(train_data.shape[2],), name="input")

# transform input signal to spikes using trainable off-chip layer
to_spikes_layer = Dense(1024, activation=tf.nn.relu, use_bias=False, )
to_spikes = to_spikes_layer(inp)

# on-chip layers
L1_layer = Dense(512, activation=tf.nn.relu, use_bias=False, )
L1 = L1_layer(to_spikes)

L2_layer = Dense(128, activation=tf.nn.relu, use_bias=False, )
L2 = L2_layer(L1)

L3_layer = Dense(64, activation=tf.nn.relu, use_bias=False, )
L3 = L3_layer(L2)

# since this final output layer has no activation function,
# it will be converted to a `nengo.Node` and run off-chip
output = Dense(units=2, name="output")(L3)

model = Model(inputs=inp, outputs=output)
model.summary()


def train(params_file="./keras_to_loihi_params", epochs=1, **kwargs):
    converter = nengo_dl.Converter(model, **kwargs)

    with nengo_dl.Simulator(converter.net, seed=0, minibatch_size=100) as sim:
        sim.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss={converter.outputs[output]: tf.keras.losses.MeanSquaredError()},
            metrics={converter.outputs[output]: tf.keras.metrics.MeanSquaredError()},
        )
        sim.fit(
            {converter.inputs[inp]: train_data},
            {converter.outputs[output]: train_truth},
            epochs=epochs,
        )

        # save the parameters to file
        sim.save_params(params_file)


# train this network with normal ReLU neurons
train(epochs=30, swap_activations={tf.nn.relu: nengo.RectifiedLinear()})


def run_network(
        activation,
        params_file="./keras_to_loihi_params",
        n_steps=30,
        scale_firing_rates=1,
        synapse=None,
        n_test=100,
        n_plots=1,
        plot_idx=-1
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        scale_firing_rates=scale_firing_rates,
        swap_activations={tf.nn.relu: activation},
        synapse=synapse,
    )

    print_neurons_type(nengo_converter)

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[output]

    # add probes to layers to record activity
    with nengo_converter.net:
        probes = collections.OrderedDict([[L1_layer, nengo.Probe(nengo_converter.layers[L1])],
                                          [L2_layer, nengo.Probe(nengo_converter.layers[L2])],
                                          [L3_layer, nengo.Probe(nengo_converter.layers[L3])], ])

    # repeat inputs for some number of timesteps
    tiled_test_data = np.tile(test_data[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=1, progress_bar=False
    ) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_data})

    # compute accuracy on test data, using output of network on last timestep
    test_predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    correct = test_truth[:n_test, 0, 0]
    print("Test accuracy: %.2f%%" % (100 * np.mean(test_predictions == correct)))

    predicted = np.array(test_predictions, dtype=int)
    correct = np.array(correct, dtype=int)

    # Plot normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(outdir + f'/{plot_idx}_confusion_matrix.jpg')

    # plot the results
    mean_rates = []
    for i in range(n_plots):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        # TODO: add a plot of current input signal
        # plt.title("Input signal")
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
            print('"%s" mean firing rate (example %d): %0.1f' % (layer.name, i, mean_rate))

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
mean_rates = run_network(activation=nengo.RectifiedLinear(), n_steps=10, plot_idx=plot_no)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

# test the trained networks using spiking neurons
run_network(activation=nengo.SpikingRectifiedLinear(), scale_firing_rates=100, synapse=0.005, plot_idx=plot_no)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

# test the trained networks using spiking neurons
run_network(activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(), scale_firing_rates=100, synapse=0.005,
            plot_idx=plot_no)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1


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
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

plt.figure(figsize=(10, 3))
plot_activation(nengo.LIF(), -4, 40)
plot_activation(nengo_loihi.neurons.LoihiLIF(), -4, 40)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

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
    synapse=0.005, plot_idx=plot_no
)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

# train this network with normal ReLU neurons
train(
    params_file="./keras_to_loihi_loihineuron_params",
    epochs=30,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    scale_firing_rates=100,
)

# test the trained networks using spiking neurons
run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=100,
    params_file="./keras_to_loihi_loihineuron_params",
    synapse=0.005, plot_idx=plot_no
)
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

pres_time = 0.03  # how long to present each input, in seconds
n_test = 100  # how many samples to test

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

# build network, load in trained weights, save to network
with nengo_dl.Simulator(net) as nengo_sim:
    nengo_sim.load_params("keras_to_loihi_loihineuron_params")
    nengo_sim.freeze_params(net)

with net:
    nengo_input.output = nengo.processes.PresentInput(
        test_data, presentation_time=pres_time
    )

# set off-chip layer
with net:
    nengo_loihi.add_params(net)
    net.config[nengo_converter.layers[to_spikes].ensemble].on_chip = False

# set on-chip layers
with net:
    L1_shape = L1_layer.output_shape[1:]
    net.config[nengo_converter.layers[L1].ensemble].block_shape = nengo_loihi.BlockShape((50,), L1_shape)

    L2_shape = L2_layer.output_shape[1:]
    net.config[nengo_converter.layers[L2].ensemble].block_shape = nengo_loihi.BlockShape((50,), L2_shape)

    L3_shape = L3_layer.output_shape[1:]
    net.config[nengo_converter.layers[L3].ensemble].block_shape = nengo_loihi.BlockShape((50,), L3_shape)

print_neurons_type(nengo_converter)


# build Nengo Loihi Simulator and run network
with nengo_loihi.Simulator(net) as loihi_sim:
    loihi_sim.run(n_test * pres_time)

    # get output (last timestep of each presentation period)
    pres_steps = int(round(pres_time / loihi_sim.dt))
    output = loihi_sim.data[nengo_output][pres_steps - 1:: pres_steps]

    # compute the Loihi accuracy
    loihi_predictions = np.argmax(output, axis=-1)
    correct = test_truth[:n_test, 0, 0]
    accuracy = 100 * np.mean(loihi_predictions == correct)
    print("Loihi accuracy: %.2f%%" % accuracy)

    predicted = np.array(loihi_predictions, dtype=int)
    correct = np.array(correct, dtype=int)

    print("Predicted labels: ", predicted)
    print("Correct labels: ", correct)

plt.figure(figsize=(12, 4))
timesteps = loihi_sim.trange() / loihi_sim.dt

# plot data given to the network
# TODO: add a plot of current input signal

# plot the network predictions
plt.plot(timesteps, loihi_sim.data[nengo_output])
plt.legend(["%d" % i for i in range(10)], loc="lower left")
plt.suptitle("Output predictions")
plt.xlabel("Timestep")
plt.ylabel("Probability")
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

# Plot non-normalized confusion matrix
plot_confusion_matrix(correct, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1

# Plot normalized confusion matrix
plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig(outdir + f'/{plot_no}.jpg')
plot_no += 1
