""" Anomaly Detection with Spiking Neural Networks deployed on the Loihi chip. """

import os
import h5py as h5
import nengo
import numpy as np
import nengo_dl
import nengo_loihi
import tensorflow as tf
from gwpy.timeseries import TimeSeries
import joblib
from sklearn.preprocessing import MinMaxScaler

load = h5.File('../../dataset/default_simulated.hdf', 'r')
freq = 2048
detector = 'L1'
filtered = 1
timesteps = 100
outdir = 'training_autoencoder'
os.system('mkdir -p %s' % outdir)


def filters(array, sample_frequency):
    """ Apply preprocessing such as whitening and bandpass """
    strain = TimeSeries(array, sample_rate=int(sample_frequency))
    white_data = strain.whiten(fftlength=4, fduration=4)
    bp_data = white_data.bandpass(50, 250)
    return bp_data.value


noise_samples = load['noise_samples']['%s_strain' % (str(detector).lower())][:][:]

# With LIGO simulated data, the sample isn't pre-filtered so need to filter again.
# Real data is not filtered yet.
if bool(int(filtered)):
    print('Filtering data with whitening and bandpass')
    print(f'Sample Frequency: {freq} Hz')
    x = [filters(sample, freq) for sample in noise_samples]
    print('Filtering completed')

# Normalize the data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(x)
scaler_filename = f"{outdir}/scaler_data_{detector}"
joblib.dump(scaler, scaler_filename)

# Trim dataset to be batch-friendly and reshape into timestep format
if X_train.shape[0] % timesteps != 0:
    X_train = X_train[:-1 * int(X_train.shape[0] % timesteps)]

# Reshape inputs for LSTM [samples, timesteps, features]
X = X_train.reshape(-1, timesteps, 1)
print("Training data shape:", X_train.shape)

# creating a train and test dataset
test_d = X
train_d = X

n_inputs = (X.shape[1], X.shape[2])
n_outputs = (X.shape[1], 1)
max_rate = 100
amplitude = 1/max_rate
presentation_time = 0.1

# model for Jet classification
with nengo.Network(label="Anomaly Detection") as model:
    nengo_loihi.add_params(model)
    model.config[nengo.Connection].synapse = None

    model.config[nengo.Ensemble].max_rates = nengo.dists.Choice([max_rate])
    model.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])

    """ Choose a type of spiking neuron """
    neuron_type = nengo.SpikingRectifiedLinear(amplitude=amplitude)
    # neuron_type = nengo.LIF(tau_rc=0.02, tau_ref=0.001, amplitude=amplitude)
    # neuron_type = nengo.AdaptiveLIF(amplitude=amplitude)
    # neuron_type = nengo.Izhikevich()

    inp = nengo.Node(nengo.processes.PresentInput(test_d[0], presentation_time), size_out=n_inputs)

    out = nengo.Node(size_in=n_outputs)

    layer_1 = nengo.Ensemble(n_neurons=64, dimensions=1, neuron_type=neuron_type, label="Layer 1")
    model.config[layer_1].on_chip = False
    nengo.Connection(inp, layer_1.neurons, transform=nengo_dl.dists.Glorot())
    p1 = nengo.Probe(layer_1.neurons)

    layer_2 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 2")
    nengo.Connection(layer_1.neurons, layer_2.neurons, transform=nengo_dl.dists.Glorot())
    p2 = nengo.Probe(layer_2.neurons)

    layer_3 = nengo.Ensemble(n_neurons=32, dimensions=1, neuron_type=neuron_type, label="Layer 3")
    nengo.Connection(layer_2.neurons, layer_3.neurons, transform=nengo_dl.dists.Glorot())
    p3 = nengo.Probe(layer_3.neurons)

    nengo.Connection(layer_3.neurons, out, transform=nengo_dl.dists.Glorot())

    out_p = nengo.Probe(out)
    out_p_filt = nengo.Probe(out, synapse=nengo.Alpha(0.01))


def crossentropy(outputs, targets):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputs, labels=targets))


def classification_error(outputs, targets):
    return 100 * tf.reduce_mean(
        tf.cast(tf.not_equal(tf.argmax(outputs[:, -1], axis=-1), tf.argmax(targets[:, -1], axis=-1)), tf.float32))


dt = 0.001  # simulation timestep
step = int(presentation_time / dt)
presentation_time = 0.1  # input presentation time
train_data = {inp: train_d[0][:, None, :], out_p: train_d[1][:, None, :]}

# for the test data evaluation we will be running the network over time
# using spiking neurons, so we need to repeat the input/target data
# for a number of timesteps (based on the presentation_time)
minibatch_size = 200
test_data = {inp: np.tile(test_d[0][:minibatch_size*2, None, :], (1, step, 1)),
             out_p_filt: np.tile(test_d[1][:minibatch_size*2, None, :], (1, step, 1))}


do_training = True
with nengo_dl.Simulator(model, minibatch_size=minibatch_size, seed=0) as sim:
    if do_training:
        output = {out_p_filt: classification_error}
        loss = sim.loss(test_data, output)
        print("error before training: %.2f%%" % loss)
        # run training
        sim.train(train_data, tf.train.RMSPropOptimizer(learning_rate=0.001),
                  objective={out_p: crossentropy}, n_epochs=50)
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.save_params("./anomaly_detection_params")
    else:
        print("error before training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))
        sim.load_params("./model_files/anomaly_detection_file.ckpt")
        print("parameters loaded")
        print("error after training: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

    sim.run_steps(int(presentation_time / dt), data={inp: test_data[inp][:minibatch_size]})
    sim.freeze_params(model)

for conn in model.all_connections:
    conn.synapse = 0.005

if do_training:
    with nengo_dl.Simulator(model, minibatch_size=minibatch_size) as sim:
        print("error w/ synapse: %.2f%%" % sim.loss(test_data, {out_p_filt: classification_error}))

n_presentations = 50
with nengo_loihi.Simulator(model, dt=dt, precompute=False) as sim:
    # if running on Loihi, increase the max input spikes per step
    if 'loihi' in sim.sims:
        sim.sims['loihi'].snip_max_spikes_per_step = 120

    # run the simulation on Loihi
    sim.run(n_presentations * presentation_time)

    # check the error
    step = int(presentation_time / dt)
    output = sim.data[out_p_filt][step - 1::step]

    error_percentage = 100 * (np.mean(np.argmax(output, axis=-1) !=
                             np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1)))

    predicted = np.argmax(output, axis=-1)
    correct = np.argmax(test_data[out_p_filt][:n_presentations, -1], axis=-1)

    predicted = np.array(predicted, dtype=int)
    correct = np.array(correct, dtype=int)

    print("Predicted labels: ", predicted)
    print("Correct labels: ", correct)
    print("loihi error: %.2f%%" % error_percentage)

    np.set_printoptions(precision=2)
