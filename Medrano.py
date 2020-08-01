#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 21/08/2020
#LABORATORY 3: MODULACIÓN AM Y FM

from scipy.io import wavfile
import os.path
import numpy as np
import matplotlib.pyplot as plt
import time

# CONSTANTS 
# GLOBAL VARIABLES
# CLASSES
# FUNCTIONS
def read_wav_file(wav_filename):
    """ Read a WAV file.
    
    Return a tuple with sample rate (Hz) and data from a WAV file

    Parameters:
    ----------
    wav_filename : string
        Input wav filename. 

    Returns:
    -------
    wav_file_data: tuple
        Tuple with sample rate and sample data from a WAV file.
   """
    sample_rate, sample_data = wavfile.read(wav_filename)
    return (sample_rate, sample_data)

def is_valid_audio_file(filename):
    """ Check if it's a valid audio file.
    
    Returns True if it's valid, False if it's not.
    
    Parameters:
    ----------
    filename : string
        Input audio filename. 

    Returns:
    -------
    Status: boolean
        True if it's valid, False if it's not.
   """
    if not os.path.exists(filename):
        return False
    elif filename[-4:] != ".wav":
        return False
    else:
        return True

def get_time_audio_signal(fs, wave):
    """ Get time range in a wave, based on the frequency.
    
    Return the wave time range
    
    Parameters:
    ----------
    fs : int
        Sample rate of wave. 
    wave : numpy array
        Sample data of wave
    Returns:
    -------
    ndarray: ndarray
        Array of evenly spaced values with the time range.
   """
    return np.arange(wave.size)/float(fs)

    # MAIN
def main():
    """ Main function of program """
    filename = "handel.wav"
    #filename = input("Choose an audio file (.wav) to read: ")
    if is_valid_audio_file(filename):
         # Reading the audio file
        fs, wave = wavfile.read(filename)
        time = get_time_audio_signal(fs, wave)
        print(fs)
        print(wave)
        print(time)

main()
#REFERENCES
#https://medium.com/@nabanita.sarkar/simulating-amplitude-modulation-using-python-6ed03eb4e712