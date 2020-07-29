import collections
import warnings

import matplotlib.pyplot as plt
import nengo
import nengo_dl
import numpy as np
import tensorflow as tf
import nengo_loihi

# ignore NengoDL warning about no GPU
warnings.filterwarnings("ignore", message="No GPU", module="nengo_dl")

# The results in this notebook should be reproducible across many random seeds.
# However, some seed values may cause problems, particularly in the `to-spikes` layer
# where poor initialization can result in no information being sent to the chip. We set
# the seed to ensure that good results are reproducible without having to re-train.
np.random.seed(0)
tf.random.set_seed(0)

# load in MNIST dataset
(
    (train_images, train_labels),
    (test_images, test_labels),
) = tf.keras.datasets.mnist.load_data()

# flatten images and add time dimension
train_images = train_images.reshape((train_images.shape[0], 1, -1))
train_labels = train_labels.reshape((train_labels.shape[0], 1, -1))
test_images = test_images.reshape((test_images.shape[0], 1, -1))
test_labels = test_labels.reshape((test_labels.shape[0], 1, -1))

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(np.reshape(train_images[i], (28, 28)), cmap="gray")
    plt.axis("off")
    plt.title(str(train_labels[i, 0, 0]))

inp = tf.keras.Input(shape=(28, 28, 1), name="input")

# transform input signal to spikes using trainable 1x1 convolutional layer
to_spikes_layer = tf.keras.layers.Conv2D(
    filters=3,  # 3 neurons per pixel
    kernel_size=1,
    strides=1,
    activation=tf.nn.relu,
    use_bias=False,
    name="to-spikes",
)
to_spikes = to_spikes_layer(inp)

# on-chip convolutional layers
conv0_layer = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
    use_bias=False,
    name="conv0",
)
conv0 = conv0_layer(to_spikes)

conv1_layer = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
    use_bias=False,
    name="conv1",
)
conv1 = conv1_layer(conv0)

flatten = tf.keras.layers.Flatten(name="flatten")(conv1)

dense0_layer = tf.keras.layers.Dense(units=100, activation=tf.nn.relu, name="dense0")
dense0 = dense0_layer(flatten)

# since this final output layer has no activation function,
# it will be converted to a `nengo.Node` and run off-chip
dense1 = tf.keras.layers.Dense(units=10, name="dense1")(dense0)

model = tf.keras.Model(inputs=inp, outputs=dense1)
model.summary()


def train(params_file="./keras_to_loihi_params", epochs=1, **kwargs):
    converter = nengo_dl.Converter(model, **kwargs)

    with nengo_dl.Simulator(converter.net, seed=0, minibatch_size=200) as sim:
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={
                converter.outputs[dense1]: tf.losses.SparseCategoricalCrossentropy(
                    from_logits=True
                )
            },
            metrics={converter.outputs[dense1]: tf.metrics.sparse_categorical_accuracy},
        )
        sim.fit(
            {converter.inputs[inp]: train_images},
            {converter.outputs[dense1]: train_labels},
            epochs=epochs,
        )

        # save the parameters to file
        sim.save_params(params_file)


# train this network with normal ReLU neurons
train(
    epochs=2, swap_activations={tf.nn.relu: nengo.RectifiedLinear()},
)


def run_network(
        activation,
        params_file="./keras_to_loihi_params",
        n_steps=30,
        scale_firing_rates=1,
        synapse=None,
        n_test=100,
        n_plots=2,
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
    nengo_output = nengo_converter.outputs[dense1]

    # add probes to layers to record activity
    with nengo_converter.net:
        probes = collections.OrderedDict(
            [
                [to_spikes_layer, nengo.Probe(nengo_converter.layers[to_spikes])],
                [conv0_layer, nengo.Probe(nengo_converter.layers[conv0])],
                [conv1_layer, nengo.Probe(nengo_converter.layers[conv1])],
                [dense0_layer, nengo.Probe(nengo_converter.layers[dense0])],
            ]
        )

    # repeat inputs for some number of timesteps
    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=20, progress_bar=False
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

        plt.subplot(1, 3, 1)
        plt.title("Input image")
        plt.imshow(test_images[i, 0].reshape((28, 28)), cmap="gray")
        plt.axis("off")

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
    to_spikes_layer: target_mean / mean_rates[0],
    conv0_layer: target_mean / mean_rates[1],
    conv1_layer: target_mean / mean_rates[2],
    dense0_layer: target_mean / mean_rates[3],
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
nengo_output = nengo_converter.outputs[dense1]

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
    net.config[nengo_converter.layers[to_spikes].ensemble].on_chip = False

with net:
    conv0_shape = conv0_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv0].ensemble
    ].block_shape = nengo_loihi.BlockShape((16, 16, 4), conv0_shape)

    conv1_shape = conv1_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[conv1].ensemble
    ].block_shape = nengo_loihi.BlockShape((8, 8, 16), conv1_shape)

    dense0_shape = dense0_layer.output_shape[1:]
    net.config[
        nengo_converter.layers[dense0].ensemble
    ].block_shape = nengo_loihi.BlockShape((50,), dense0_shape)

# build Nengo Loihi Simulator and run network
with nengo_loihi.Simulator(net) as loihi_sim:
    loihi_sim.run(n_test * pres_time)

    # get output (last timestep of each presentation period)
    pres_steps = int(round(pres_time / loihi_sim.dt))
    output = loihi_sim.data[nengo_output][pres_steps - 1:: pres_steps]

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
    allimage[:, i * nj: (i + 1) * nj] = image
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
