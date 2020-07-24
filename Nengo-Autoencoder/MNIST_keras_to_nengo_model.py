""" Example MNIST code for Converting a Keras model to a spiking neural network """

from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import nengo
import numpy as np
import tensorflow as tf
import nengo_dl

seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

(train_images, train_labels), (test_images, test_labels) = (tf.keras.datasets.mnist.load_data())

# flatten images and add time dimension
train_images = train_images.reshape((train_images.shape[0], 1, -1))
train_labels = train_labels.reshape((train_labels.shape[0], 1, -1))
test_images = test_images.reshape((test_images.shape[0], 1, -1))
test_labels = test_labels.reshape((test_labels.shape[0], 1, -1))

plt.figure(figsize=(12, 4))
for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.imshow(np.reshape(train_images[i], (28, 28)), cmap="gray")
    plt.axis('off')
    plt.title(str(train_labels[i, 0, 0]));

# input
inp = tf.keras.Input(shape=(28, 28, 1))

# convolutional layers
conv0 = tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=3,
    activation=tf.nn.relu,
)(inp)

conv1 = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    strides=2,
    activation=tf.nn.relu,
)(conv0)

# fully connected layer
flatten = tf.keras.layers.Flatten()(conv1)
dense = tf.keras.layers.Dense(units=10)(flatten)

model = tf.keras.Model(inputs=inp, outputs=dense)

converter = nengo_dl.Converter(model)

do_training = False
if do_training:
    with nengo_dl.Simulator(converter.net, minibatch_size=200) as sim:
        # run training
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],
        )
        sim.fit(
            {converter.inputs[inp]: train_images},
            {converter.outputs[dense]: train_labels},
            validation_data=(
                {converter.inputs[inp]: test_images},
                {converter.outputs[dense]: test_labels},
            ),
            epochs=2,
        )

        # save the parameters to file
        sim.save_params("./keras_to_snn_params")
else:
    # download pretrained weights
    urlretrieve(
        "https://drive.google.com/uc?export=download&"
        "id=1lBkR968AQo__t8sMMeDYGTQpBJZIs2_T",
        "keras_to_snn_params.npz")
    print("Loaded pretrained weights")


def run_network(activation, params_file="keras_to_snn_params", n_steps=30,
                scale_firing_rates=1, synapse=None, n_test=400):
    # convert the keras model to a nengo network
    nengo_converter = nengo_dl.Converter(
        model,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )

    # get input/output objects
    nengo_input = nengo_converter.inputs[inp]
    nengo_output = nengo_converter.outputs[dense]

    # add a probe to the first convolutional layer to record activity
    with nengo_converter.net:
        conv0_probe = nengo.Probe(nengo_converter.layers[conv0])

    # repeat inputs for some number of timesteps
    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(stateful=False)

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
            nengo_converter.net, minibatch_size=10,
            progress_bar=False) as nengo_sim:
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})

    # compute accuracy on test data, using output of network on
    # last timestep
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
    print("Test accuracy: %.2f%%" % (100 * accuracy))

    # plot the results
    for ii in range(4):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.title("Input image")
        plt.imshow(test_images[ii, 0].reshape((28, 28)), cmap="gray")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        sample_neurons = np.linspace(
            0,
            data[conv0_probe].shape[-1],
            1000,
            endpoint=False,
            dtype=np.int32,
        )
        scaled_data = data[conv0_probe][ii, :, sample_neurons].T * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_data *= 0.001
            rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel('Number of spikes')
        else:
            rates = scaled_data
            plt.ylabel('Firing rates (Hz)')
        plt.xlabel('Timestep')
        plt.title(
            "Neural activities (conv0 mean=%dHz max=%dHz)" % (
                rates.mean(), rates.max())
        )
        plt.plot(scaled_data)

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(tf.nn.softmax(data[nengo_output][ii]))
        plt.legend([str(j) for j in range(10)], loc="upper left")
        plt.xlabel('Timestep')
        plt.ylabel("Probability")

        plt.tight_layout()


run_network(activation=nengo.RectifiedLinear(), n_steps=10)
run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=10)

for s in [0.001, 0.005, 0.01]:
    print("Synapse=%.3f" % s)
    run_network(activation=nengo.SpikingRectifiedLinear(), n_steps=120, synapse=s)
    plt.show()

for scale in [2, 10, 20]:
    print("Scale=%d" % scale)
    run_network(activation=nengo.SpikingRectifiedLinear(), scale_firing_rates=scale, synapse=0.01)
    plt.show()
