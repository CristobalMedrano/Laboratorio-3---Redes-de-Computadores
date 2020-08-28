#-*- coding: utf-8 -*-
#AUTHOR: Cristóbal Nicolás Medrano Alvarado (19.083.864-1)
#DATE: 21/08/2020
#LABORATORY 3: MODULACIÓN ANALÓGICA

from scipy.io import wavfile
from scipy.integrate import quad
from scipy import interpolate
from numpy import pi, cos, arange, fft, linspace
import os.path
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

def get_time_audio_signal(fs, signal):
    """ Get time range in a signal, based on the frequency.
    
    Return the signal time range
    
    Parameters:
    ----------
    fs : int
        Sample rate of signal. 
    signal : numpy array
        Sample data of signal
    Returns:
    -------
    ndarray: ndarray
        Array of evenly spaced values with the time range.
   """
    return arange(signal.size)/float(fs)

def am_carrier_signal(fc, t):
    """ Get amplitude modulation (AM) simulation
    
    Return amplitude modulation (AM)
    
    Parameters:
    ----------
    fc : int
        Sample rate of carrier signal in Hz. 
    t : ndarray
        Sample time of signal
    Returns:
    -------
    am_carrier_time: ndarray
        Array of evenly spaced values with the time range modulation.
    am_carrier_signal: ndarray
        Array of evenly spaced values with the time range modulation.
    am_carrier_sample_rate: ndarray
        Array of evenly spaced values with the time range modulation.
    """
    min_time = 0
    max_time = t[-1]
    carrier_sample_rate = 4 * fc
    carrier_time = arange(min_time, max_time, 1/carrier_sample_rate)
    carrier_signal = cos(( 2 * pi * fc ) * carrier_time)

    return carrier_time, carrier_signal, carrier_sample_rate

def simulate_amplitude_modulation(k, m, fc, t):
    """ Get amplitude modulation (AM) simulation
    
    Return amplitude modulation (AM)
    
    Parameters:
    ----------
    k: int
        Modulation index or frequency deviation
    m: tuple
        Message to modulate (fs and signal in this project)
    fc : int
        Sample rate of carrier signal in Hz. 
    t : ndarray
        Sample time of signal
    Returns:
    -------
    am: ndarray
        Array of evenly spaced values with the time range modulation.
    """    
    min_time = 0
    max_time = t[-1]
    
    fs = m[0]
    signal = m[1]

    # Carrier Signal
    carrier_time, carrier_signal, carrier_sample_rate = am_carrier_signal(fc, t)
    
    # Plot the carrier signal
    plot_signal(carrier_time, carrier_signal, "Carrier Signal in time", "Time(s)", "Amplitude", min_time, max_time)
    
    # Get the Fourier transform of carrier signal
    carrier_signal_fft_freq, carrier_signal_fft = get_fourier_transform(carrier_sample_rate, carrier_signal)
    
    # Plot the Fourier transform of carrier fft signal
    plot_signal(carrier_signal_fft_freq, abs(carrier_signal_fft), "Carrier Signal in Frequency", "Frequency(hz)", "|F(w)|", min(carrier_signal_fft_freq), max(carrier_signal_fft_freq))

    print('Señal Portadora (Frecuencia [fc]): ', fc, 'hz.')
    print('Señal Portadora (Frecuencia de Muestreo [fsc]): ', len(carrier_time), 'hz.')
    #plt.show()

    # Interpolacion de la señal original a la frecuencia de la portadora
    print("frecuencia del audio :"+ str(fs))
    interpolated_signal = get_interpolated_signal(signal, fs)

    print("largo de la señal "+ str(len(interpolated_signal(carrier_time))))
    print("largo de la señal portadora "+ str(len(carrier_signal)))
    plot_signal(carrier_time, interpolated_signal(carrier_time), "Interpolated Signal in time", "Time(s)", "Amplitude", min_time, max_time)


    am_modulated_signal = interpolated_signal(carrier_time)*carrier_signal
    plot_signal(carrier_time, am_modulated_signal, "Modulated Signal in time", "Time(s)", "Amplitude", min_time, max_time)
    plt.show()

    return 1

def simulate_frequency_modulation(k, m, fc, t):
    """ Get frequency modulation (FM) simulation
    
    Return frequency modulation (FM)
    
    Parameters:
    ----------
    k: int
        Modulation index or frequency deviation
    m: ndarray
        Message to dispatch
    fc : int
        Sample rate of carrier signal. 
    t : ndarray
        Sample time of signal
    Returns:
    -------
    fm: ndarray
        Array of evenly spaced values with the time range modulation.
   """
    return cos((2*pi*fc)*t + k*quad(m, a=0, b=t))

def get_fourier_transform(fs, signal):
    """ Get the fourier transform of the audio signal.
    
    Return the Discrete Fourier Transform sample frequencies and the
    truncated or zero-padded input, transformed along the axis
    
    Parameters:
    ----------
    fs : int
        Sample rate of signal. 
    signal : numpy array
        Sample data of signal
    Returns:
    -------
    fft_freq: ndarray
        The Discrete Fourier Transform sample frequencies.
    fft_signal: complex ndarray
        The truncated or zero-padded input, transformed along the axis.
    """
    fft_freq = fft.fftfreq(signal.size, 1/fs)
    fft_signal = fft.fft(signal)
    return fft_freq, fft_signal

def plot_signal(x, y, title, xlabel, ylabel, min_x, max_x):
    """ Function that graphs a signal.
      
    Parameters:
    ----------
    x : numpy array
        Points of the function corresponding to the x-axis
    y : numpy array
        Points of the function corresponding to the y-axis
    title: string
        Plot title
    xlabel: string
        Plot x-axis label
    ylabel: string
        Plot y-axis label
    min_x: int
        set the x limits of the current axes (left).
    max_x: int
        set the x limits of the current axes (right).
    """
    plt.figure(title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(x, y)
    plt.xlim(min_x, max_x)
    plt.grid()

def get_interpolated_signal(signal, fs):
    """ Function that interpolates an array
      
    Parameters:
    ----------
    signal : numpy array
        Sample data of signal
    fs : int
        Sample rate of signal. 
    Returns:
    """
    t = get_time_audio_signal(fs, signal)
    return interpolate.interp1d(t, signal) 

# MAIN
def main():
    """ Main function of program """
    filename = "handel.wav"
    #filename = input("Choose an audio file (.wav) to read: ")
    print("\nLaboratorio 3 - Modulación Analógica\n(Desarrollado por: Cristóbal Medrano A.)\n")
    print("Adjunto a este programa, se encuentran 1 audio de prueba:")
    print("'handel.wav'\n")
    #filename = input("Escriba el nombre del archivo de audio a procesar: ")
    print()
    if is_valid_audio_file(filename):
        # Params
        # fs, ts, fc, tc, fm

         # Reading the audio file
        fs, signal = wavfile.read(filename)
        t = get_time_audio_signal(fs, signal)

        # Getting the fft.
        signal_fft_freq, signal_fft = get_fourier_transform(fs, signal)
        #am = simulate_amplitude_modulation(1, signal, fs, t)
        
        #Grafico y transformada
        plot_signal(t, signal, "Signal in time", "Time(s)", "Amplitude", min(t), max(t))
        plot_signal(signal_fft_freq, abs(signal_fft), "Signal in Frequency", "Frequency(hz)", "|F(w)|", min(signal_fft_freq), max(signal_fft_freq))
        #Calcular la frecuencia maxima de la señal original
        print('Señal de audio (Frecuencia de Muestreo [fs]): ', fs, 'hz.')
        print('Señal de audio (Frecuencia Máxima [fm]): ', (fs/2), 'hz.')
        # fm(frecuencia maxima de la señal) = 4096hz
        # fc >> fm >> 4096hz

        # es necesario muestrear esta frecuencia cumpliendo el teorema de nyquist
        # es decir, carrier_sample_rate >> 2fc
        
        # frecuencia de la portadora
        k = 1
        fc = 24000 # hz
        m = (fs, signal)
        simulate_amplitude_modulation(k, m, fc, t)
        '''# frecuencia de muestreo de la portadora
        carrier_sample_rate = 4*fc #3000 hz
        carrier_time = arange(0, t[-1], 1/carrier_sample_rate)
        carrier_signal = am_carrier_signal(fc, carrier_time)
        plot_signal(carrier_time, carrier_signal, "Carrier Signal in time", "Time(s)", "Amplitude", 0, t[-1])
        
        carrier_signal_fft_freq, carrier_signal_fft = get_fourier_transform(carrier_sample_rate, carrier_signal)
        plot_signal(carrier_signal_fft_freq, abs(carrier_signal_fft), "Carrier Signal in Frequency", "Frequency(hz)", "|F(w)|", min(carrier_signal_fft_freq), max(carrier_signal_fft_freq))

        print('Señal Portadora (Frecuencia [fc]): ', fc, 'hz.')
        print('Señal Portadora (Frecuencia de Muestreo [carrier_sample_rate]): ', len(carrier_time), 'hz.')
        #plt.show()

        # Interpolacion de la señal original a la frecuencia de la portadora
        interpolated_signal = get_interpolated_signal(signal, fs)

        print("largo de la señal "+ str(len(interpolated_signal(carrier_time))))
        print("largo de la señal portadora "+ str(len(carrier_signal)))
        plot_signal(carrier_time, interpolated_signal(carrier_time), "Interpolated Signal in time", "Time(s)", "Amplitude", 0, t[-1])


        am_modulated_signal = interpolated_signal(carrier_time)*carrier_signal
        plot_signal(carrier_time, am_modulated_signal, "Modulated Signal in time", "Time(s)", "Amplitude", 0, t[-1])
        plt.show()
        '''
        # Transformada de Fourier de la señal interpolada
        # Transformada de Fourier de la señal Modulada

        # Demodulacion
        # Transformada de Fourier de la señal Demodulada
        # Aplicacion del filtro pasa bajos
        # Transformada de Fourier de la señal demodulada luego de filtro

        # Modulacion FM
        # Grafico en el tiempo FM
        # Transformada de Fourier FM

        #A partir de lo anterior calcular la fc y tc de la portadora
        #Grafico y transformada
        
        #fc = fs/4 # hz

        #plot_signal(t, am_carrier_signal(fc, t), "Carrier Signal in time", "time (s)", "amplitude", 0, 0.01)

        #am_carrier_fft_freq, am_carrier_fft_signal = get_fourier_transform(fc, am_carrier_signal(fc, t))
        #plot_signal(am_carrier_fft_freq, abs(am_carrier_fft_signal), "Carrier Signal in frequency", 'Frequency (Hz)', '|F(w)|', -fc/2, fc/2)
        #plt.show()
        #f_carrier = am_carrier_signal(fc, t)
        #f_carrier_wave_fft = abs(fft.fft(f_carrier))
        #f_carrier_fs_fft = fft.fftfreq(f_carrier_wave_fft.size) 
        #plot_signal(f_carrier_fs_fft, f_carrier_wave_fft, "Carrier Signal in frequency", "frequency (hz)", "|f(w)|", min(f_carrier_fs_fft), max(f_carrier_fs_fft))
        #plt.show()
        #plot_signal(time, am_carrier_signal(fs, time), "Signal in time carrier", "time (s)", "amplitude", min(time), max(time))
        #print(am)
        #print(fs)
        #print(signal)
        #print(t)
    else: 
        print("El archivo ingresado no existe.")

main()
#REFERENCES
#https://medium.com/@nabanita.sarkar/simulating-amplitude-modulation-using-python-6ed03eb4e712
#https://uvirtual.usach.cl/moodle/pluginfile.php/312438/mod_resource/content/0/Laboratorio%203%20-%20Modulaci%C3%B3n%20Anal%C3%B3gica.pdf