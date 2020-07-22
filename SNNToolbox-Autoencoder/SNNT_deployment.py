""" Script to run a pre-trained neural network for anomaly detection on Loihi, using SNN toolbox  """

import argparse
from snntoolbox.bin.run import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config_filepath = 'conversion_config_loihi.txt'
    parser.add_argument("--model", help="Path to model with weights in h5 format",
                        action='store', dest='model', default=config_filepath)
    args = parser.parse_args()
    main(args.model)
