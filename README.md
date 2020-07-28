LIGO-Autoencoder: An anomaly detection algorithm for gravitational waves
======================================================================================

Setup
======================================================================================
Download dataset containing 5,000 injection events and 20,000 noise events. Place in directory labeled 'data'. Must have GWpy package. 

The data used in this package was produced from https://github.com/timothygebhard/ggwd 

Simulated data: https://cernbox.cern.ch/index.php/s/QMViw5nZZfbrsf9
  - Unfiltered 
  - Sample Rate: 2048 (put 2 KHz as sampling frequency)
  - Generates simulated noise and simulated waveforms 

Training
======================================================================================

Determine the parameters needed for the autoencoder training. For example: 

  - Output directory = training_LSTM_100steps
  - Detector = L1 (L1 and H1 are available)
  - Sampling Frequency = 2 KHz (used for filters, input in KHz)
  - Filters = 1 (turn on for simulated data)
  - Timesteps = 100 

Would be executed by running: 

```bash
python3 train.py training_LSTM_100steps L1 --freq 2 --filtered 1 --timesteps 100
```
Batch size and model achitecture need to be changed manually inside the train.py script. Training only occurs with noise events.


Testing
======================================================================================
The testing procedure involves determining a upper-threshold for the loss from the training data. Then, the loss plots of 10 random gravitational waves are plotted. This can be changed in the code. 
Run the evaluation script using the same parameters used in training: 

```bash
python3 eval.py training_LSTM_100steps L1 --freq 2 --filtered 1 --timesteps 100
```

Additional Models
======================================================================================
Add additional models in models.py and import into train.py
