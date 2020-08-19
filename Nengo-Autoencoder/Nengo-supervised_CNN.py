""" Supervised anomaly detection in LIGO data with Spiking Neural Networks running on Loihi """

import os
import collections
import warnings

import nengo
import nengo_dl
import nengo_loihi
import numpy as np
import tensorflow as tf
import h5py as h5
import matplotlib.pyplot as plt
from gwpy.timeseries import TimeSeries
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Reshape, Conv2D, Flatten
from keras.models import Model


def plot_roc_corve(y_true, y_pred):
    """ This function plots the ROC curve. """
    plt.figure()
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, lw=2, label='%s (auc = %0.2f)' % ('Nengo Loihi', auc(fpr, tpr)))
    plt.xlim([1e-4, 1])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.xscale('log')
    plt.title('LIGO Supervised GW-Detection')
    plt.legend(loc="lower right")


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

# Create output directory where the results will be saved
outdir = "Outputs"
os.system('mkdir ' + outdir)

# Counter of plots to save
plot_no = 0

# Load train and test data
load = h5.File('../../dataset/240k_1sec_L1.h5', 'r')
X_train = load['data'][:]

datapoints = 120000
gw = np.concatenate((np.ones(datapoints), np.zeros(datapoints)))
noise = np.concatenate((np.zeros(datapoints), np.ones(datapoints)))
targets = np.transpose(np.array([gw, noise]))

# Splitting the train / test data in ratio 80:20
train_data, test_data, train_truth, test_truth = train_test_split(X_train, targets, test_size=0.2, random_state=42)
class_names = np.array(['noise', 'GW'], dtype=str)
del X_train, targets

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

x = Reshape((train_data.shape[2], 1, 1))(inp)

# transform input signal to spikes using trainable off-chip layer
to_spikes_layer = Conv2D(16, (4, 1), activation=tf.nn.relu, use_bias=False)
to_spikes = to_spikes_layer(x)

# on-chip layers
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

# since this final output layer has no activation function, it will be converted to a `nengo.Node` and run off-chip
output = Dense(units=2, name="output")(L6)

model = Model(inputs=inp, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
model.summary()

# train the model in clear keras code
"""
history = model.fit(train_data, train_truth, epochs=1, batch_size=16, validation_split=0.2,).history
"""
del train_data, train_truth


def train(params_file="./keras_to_loihi_params", epochs=1, **kwargs):
    converter = nengo_dl.Converter(model, max_to_avg_pool=True, **kwargs)

    print_neurons_type(converter)

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


# train the model in Nengo with normal ReLU neurons
"""
train(epochs=15, params_file="./keras_to_loihi_params_15epoch", swap_activations={tf.nn.relu: nengo.RectifiedLinear()})
"""


def run_network(
        activation,
        params_file="./keras_to_loihi_params_15epoch",
        n_steps=30,
        scale_firing_rates=1,
        synapse=None,
        n_test=1000,
        n_plots=1,
        plot_idx=-1
):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        scale_firing_rates=scale_firing_rates,
        swap_activations={tf.nn.relu: activation},
        synapse=synapse,
        max_to_avg_pool=True
    )

    # set a low-pass filter value on all synapses in the network
    if synapse is not None:
        for conn in nengo_converter.net.all_connections:
            conn.synapse = synapse

    print_neurons_type(nengo_converter)

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[output]

    # add probes to layers to record activity
    with nengo_converter.net:
        probes = collections.OrderedDict([[L1_layer, nengo.Probe(nengo_converter.layers[L1])],
                                          [L2_layer, nengo.Probe(nengo_converter.layers[L2])],
                                          [L3_layer, nengo.Probe(nengo_converter.layers[L3])],
                                          [L4_layer, nengo.Probe(nengo_converter.layers[L4])],
                                          [L5_layer, nengo.Probe(nengo_converter.layers[L5])],
                                          [L6_layer, nengo.Probe(nengo_converter.layers[L6])]])

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

    plot_roc_corve(correct, predicted)
    plt.savefig(outdir + '/%s_ROC_curve_evaluation.jpg' % plot_idx)
    plot_idx += 1

    # Plot normalized confusion matrix
    plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                          title='Normalized confusion matrix')
    plt.savefig(outdir + '/%s_confusion_matrix_evaluation.jpg' % plot_idx)

    # plot the results
    mean_rates = []
    for i in range(n_plots):
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 3, 1)
        if test_truth[i][0][0] == 1:
            plot_label = 'gravitational wave'
        else:
            plot_label = 'noise'
        plt.title('Sample number: %s\nLabel: %s' % (i, plot_label))
        plt.xlabel("Timestep")
        plt.plot(test_data[i][0], color='C0')

        n_layers = len(probes)
        mean_rates_i = []
        for j, layer in enumerate(probes.keys()):
            probe = probes[layer]
            plt.subplot(n_layers, 3, (j * 3) + 2)
            outputs = data[probe][i]

            # look only at non-zero outputs
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
                ylabel = "# of Spikes"
            else:
                ylabel = "Firing rates (Hz)"

            plt.suptitle("Neural activities\n%s" % ylabel)

            # plot outputs of first 100 neurons
            plt.plot(outputs[:, :100])

        mean_rates.append(mean_rates_i)

        plt.xlabel("Timestep")
        plt.subplot(1, 3, 3)
        plt.title("Output predictions\nProbability")
        plt.plot(tf.nn.softmax(data[nengo_output][i]))
        plt.legend([str(j) for j in range(2)], loc="upper left")
        plt.xlabel("Timestep")

        plt.tight_layout()

    # take mean rates across all plotted examples
    mean_rates = np.array(mean_rates).mean(axis=0)

    return mean_rates, plot_idx


def is_spiking_type(neuron_type):
    return isinstance(neuron_type, (nengo.LIF, nengo.SpikingRectifiedLinear))


# test the trained network with normal ReLU neurons
"""
mean_rates, plot_no = run_network(activation=nengo.RectifiedLinear(), n_steps=50, plot_idx=plot_no)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
"""

# test the trained network using Nengo spiking neurons
"""
_, plot_no = run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=50, 
            plot_idx=plot_no, scale_firing_rates=5000, synapse=0.005)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
"""

# test the trained networks using Loihi spiking neurons
"""
_, plot_no = run_network(activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(), n_steps=50,
             plot_idx=plot_no, scale_firing_rates=5000, synapse=0.005)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
"""


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
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1

plt.figure(figsize=(10, 3))
plot_activation(nengo.LIF(), -4, 40)
plot_activation(nengo_loihi.neurons.LoihiLIF(), -4, 40)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1

# scale the firing rates basing on previous results
"""
target_mean = 200
scale_firing_rates = {
    L1_layer: target_mean / mean_rates[0],
    L2_layer: target_mean / mean_rates[1],
    L3_layer: target_mean / mean_rates[2],
    L4_layer: target_mean / mean_rates[3],
    L5_layer: target_mean / mean_rates[4],
    L6_layer: target_mean / mean_rates[5],
}

# test the trained networks using Loihi spiking neurons
_, plot_no = run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=scale_firing_rates,
    synapse=0.005, plot_idx=plot_no, n_steps=50
)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
"""

# train this network with Loihi spiking neurons on low firing rate
"""
train(
    params_file="./keras_to_loihi_neuron_params_15epoch",
    epochs=25,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    scale_firing_rates=100,
)
"""

# test the trained networks using Loihi spiking neurons
"""
_, plot_no = run_network(
    activation=nengo_loihi.neurons.LoihiSpikingRectifiedLinear(),
    scale_firing_rates=5000,
    params_file="./keras_to_loihi_loihineuron_params_15epoch",
    synapse=0.005, plot_idx=plot_no, n_steps=50
)
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
"""

# Prepare the model to run on-chip
pres_time = 0.06  # how long to present each input, in seconds
n_test = 200  # how many samples to test

# convert the keras model to a nengo network
nengo_converter = nengo_dl.Converter(
    model,
    scale_firing_rates=5000,
    swap_activations={tf.nn.relu: nengo_loihi.neurons.LoihiSpikingRectifiedLinear()},
    synapse=0.005,
    max_to_avg_pool=True,
)
net = nengo_converter.net

# get input/output objects
nengo_input = nengo_converter.inputs[inp]
nengo_output = nengo_converter.outputs[output]

# build network, load in trained weights, save to network
with nengo_dl.Simulator(net) as nengo_sim:
    nengo_sim.load_params("keras_to_loihi_params_15epoch")
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
    net.config[nengo_converter.layers[L1].ensemble].block_shape = nengo_loihi.BlockShape((16, 16, 4), L1_shape)

    L2_shape = L2_layer.output_shape[1:]
    net.config[nengo_converter.layers[L2].ensemble].block_shape = nengo_loihi.BlockShape((8, 8, 16), L2_shape)

    L3_shape = L3_layer.output_shape[1:]
    net.config[nengo_converter.layers[L3].ensemble].block_shape = nengo_loihi.BlockShape((4, 4, 32), L3_shape)

    L4_shape = L4_layer.output_shape[1:]
    net.config[nengo_converter.layers[L4].ensemble].block_shape = nengo_loihi.BlockShape((2, 2, 64), L4_shape)

    L5_shape = L5_layer.output_shape[1:]
    net.config[nengo_converter.layers[L5].ensemble].block_shape = nengo_loihi.BlockShape((50,), L5_shape)

    L6_shape = L6_layer.output_shape[1:]
    net.config[nengo_converter.layers[L6].ensemble].block_shape = nengo_loihi.BlockShape((50,), L6_shape)

print_neurons_type(nengo_converter)

# build Nengo Loihi Simulator and run network in simulation or on Intel Loihi Hardware
with nengo_loihi.Simulator(net, remove_passthrough=False) as loihi_sim:
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

    plot_roc_corve(correct, predicted)
    plt.savefig(outdir + '/ROC_curve_%s.jpg' % plot_no)
    plot_no += 1

timesteps = loihi_sim.trange() / loihi_sim.dt

# plot data given to the network
plt.figure(figsize=(12, 4))
plt.subplot(2, 1, 1)
plt.plot(test_data[:n_test][0], color='C0')

# plot the network predictions
plt.subplot(2, 1, 2)
plt.plot(timesteps, loihi_sim.data[nengo_output])
plt.legend([str(j) for j in range(2)], loc="upper left")
plt.suptitle("Output predictions")
plt.xlabel("Timestep")
plt.ylabel("Probability")
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1

# Plot non-normalized confusion matrix
plot_confusion_matrix(correct, predicted, classes=class_names,
                      title='Confusion matrix, without normalization')
plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1

# Plot normalized confusion matrix
plot_confusion_matrix(correct, predicted, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.savefig(outdir + '/%s.jpg' % plot_no)
plot_no += 1
