# -*- coding: utf-8 -*-
"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

#!/usr/bin/env python3
The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
from ml import train_svmm_model, predict_with_svmm_model

import sys
import pandas as pd
import os
import argparse
parser = argparse.ArgumentParser(description='Know which condition state the user is in.')
parser.add_argument('--mode', type=str, default='', help='The mode of the program. Either collecting or predicting.')
parser.add_argument('--image', type=str, default='', help='The name of the image to show')
parser.add_argument('--trial', type=str, default='', help='The trial number')
args = parser.parse_args()

# Handy little enum to make code more readable

def most_frequent(List):
    tuple_list = [tuple(item) for item in List]
    return max(set(tuple_list), key=tuple_list.count)

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """
# Modify these to change aspects of the signal processing

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 10

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0, 1, 2, 3]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 4))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        counter = 0 #counter for collecting data
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            guesses = ["","","","","","","","","",""]
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            for i in range(10):
                eeg_data, timestamp = inlet.pull_chunk(
                    timeout=1, max_samples=int(SHIFT_LENGTH * fs))
                
                # Only keep the channel we're interested in
                ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
                
                # Update EEG buffer with the new data
                eeg_buffer, filter_state = utils.update_buffer(
                    eeg_buffer, ch_data, notch=True,
                    filter_state=filter_state)

                """ 3.2 COMPUTE BAND POWERS """
                # Get newest samples from the buffer
                data_epoch = utils.get_last_data(eeg_buffer,
                                                EPOCH_LENGTH * fs)

                # Compute band powers
                band_powers = utils.compute_band_powers(data_epoch, fs).reshape(-1, 4)
                band_buffer, _ = utils.update_buffer(band_buffer,
                                                            band_powers)
                # Compute the average band powers for all epochs in buffer
                # This helps to smooth out noise
                smooth_band_powers = np.mean(band_buffer, axis=0)
                betaWaves = band_powers[Band.Beta]
                alphaWaves = band_powers[Band.Alpha]
                thetaWaves = band_powers[Band.Theta]
                deltaWaves = band_powers[Band.Delta]
                features_for_model = [[betaWaves, alphaWaves, thetaWaves, deltaWaves]]
                if args.mode == "predict" and args.trial != "":
                    if 'svmm_model' not in globals():  
                        svmm_model, scaler = train_svmm_model(args.trial)
                    if 'svmm_model' in globals():
                        prediction = predict_with_svmm_model(svmm_model, scaler, features_for_model)
                        guesses[i]=prediction
                
                # print("{}".format(betaWaves)) 
            if args.mode == "collect":
                
                #The section below is for collecting purposes
                print(band_powers[Band.Beta])
                print(band_powers[Band.Alpha])
                print(band_powers[Band.Theta])
                print(band_powers[Band.Delta])
                print() 

                
                # Store them in the csv file
                image_show = args.image
                trial = args.trial
                
                # Create a DataFrame
                df = pd.DataFrame({
                    'Image': [image_show],
                    'Beta_TP9': [band_powers[Band.Beta][0]],
                    'Beta_AF7': [band_powers[Band.Beta][1]],
                    'Beta_AF8': [band_powers[Band.Beta][2]],
                    'Beta_TP10': [band_powers[Band.Beta][3]],
                    'Alpha_TP9': [band_powers[Band.Alpha][0]],
                    'Alpha_AF7': [band_powers[Band.Alpha][1]],
                    'Alpha_AF8': [band_powers[Band.Alpha][2]],
                    'Alpha_TP10': [band_powers[Band.Alpha][3]],
                    'Theta_TP9': [band_powers[Band.Theta][0]],
                    'Theta_AF7': [band_powers[Band.Theta][1]],
                    'Theta_AF8': [band_powers[Band.Theta][2]],
                    'Theta_TP10': [band_powers[Band.Theta][3]],
                    'Delta_TP9': [band_powers[Band.Delta][0]],
                    'Delta_AF7': [band_powers[Band.Delta][1]],
                    'Delta_AF8': [band_powers[Band.Delta][2]],
                    'Delta_TP10': [band_powers[Band.Delta][3]],
                })
                
                
                # Store DataFrame into CSV file
                # Check if the file exists
                if not os.path.exists(f'dataset/{trial}.csv'):
                    # If file does not exist, write header
                    df.to_csv(f'dataset/{trial}.csv', mode='a', index=False, header=True)
                else:
                    # If file exists, do not write header
                    df.to_csv(f'dataset/{trial}.csv', mode='a', index=False, header=False)
                
                counter += 1 #counter 
                if counter>=50:
                    exit()
            elif args.mode == "predict":
                #print(most_frequent(guesses))
            
                if most_frequent(guesses)==('Arithmetic',):
                    print("A")
                elif most_frequent(guesses)==('Bouncy',):
                    print("B")
                elif most_frequent(guesses)==('Catch',):
                    print("C")
                elif most_frequent(guesses)==('Deny',):
                    print("D")
                elif most_frequent(guesses)==('Eat',):
                    print("E")
                elif most_frequent(guesses)==('Fight',):
                    print("F")
                else:
                    print("Error")
            else:
                print("Invalid mode. Not collect or predict.")
                sys.exit()


                #TEST FOR DELTA, THEN GAMMA, THEN BETA
                # should i do both delta and alpha?? or delta and gamma?? see if each differs

                # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
                #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
                # These metrics could also be used to drive brain-computer interfaces

                # Alpha Protocol:
                # Simple redout of alpha power, divided by delta waves in order to rule out noise
            # alpha_metric = smooth_band_powers[Band.Alpha] / \
                    #smooth_band_powers[Band.Delta]
                #print('Alpha Relaxation: ', alpha_metric)

                # Beta Protocol:
                # Beta waves have been used as a measure of mental activity and concentration
                # This beta over theta ratio is commonly used as neurofeedback for ADHD
                # beta_metric = smooth_band_powers[Band.Beta] / \
                #     smooth_band_powers[Band.Theta]
                # print('Beta Concentration: ', beta_metric)

                # Alpha/Theta Protocol:
                # This is another popular neurofeedback metric for stress reduction
                # Higher theta over alpha is supposedly associated with reduced anxiety
                # theta_metric = smooth_band_powers[Band.Theta] / \
                #     smooth_band_powers[Band.Alpha]
                # print('Theta Relaxation: ', theta_metric)

    except KeyboardInterrupt:
        print('Closing!')
