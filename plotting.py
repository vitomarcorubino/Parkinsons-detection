import numpy as np
import matplotlib.pyplot as plt
import textwrap
from scipy.io.wavfile import read
import librosa
import librosa.display


def plot_time_frequency_heatmap(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the STFT of the audio signal
    D = librosa.stft(y)

    # Convert the amplitude of the STFT to decibels
    D_db = librosa.amplitude_to_db(abs(D), ref=np.max)

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Display the spectrogram as a heatmap
    librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='log')

    # Add a colorbar to the plot
    plt.colorbar(format='%+2.0f dB')

    # Set the label of the colorbar
    plt.colorbar(format='%+2.0f dB').set_label('Amplitude (dB)')

    # Set the title of the plot
    plt.title('Time-Frequency Representation')

    # Set the x-label of the plot
    plt.xlabel('Time')

    # Set the y-label of the plot
    plt.ylabel('Frequency')

    # Display the plot
    plt.show()


def plot_trimmed_audio(audio_path, start_times, end_times, number_of_words, words):
    """
    This function plots the waveform of an audio file and marks the start and end times of each trimmed segment.
    It also displays the words corresponding to each segment in the plot.

    Args:
        audio_path (str): The path to the audio file that needs to be plotted.
        start_times (list): A list of start times for each trimmed segment.
        end_times (list): A list of end times for each trimmed segment.
        number_of_words (int): The number of words in each segment.
        words (list): A list of words corresponding to each segment.
    """
    # Load audio file
    sample_rate, audio_data = read(audio_path)

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Create time array
    time = np.arange(0, len(audio_data)) / sample_rate

    # Plot original audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(time, audio_data, label='Original Audio')

    # Plot red vertical line for each start and end time of trimming
    for i in range(0, len(start_times), number_of_words):
        plt.axvline(x=start_times[i], color='r')  # Start of trimming
        end_index = i + number_of_words - 1 if i + number_of_words - 1 < len(end_times) else len(end_times) - 1
        plt.axvline(x=end_times[end_index], color='r')  # End of trimming

        # Calculate the middle of the segment
        middle_time = (start_times[i] + end_times[end_index]) / 2

        # Get the corresponding words
        segment_words = ' '.join(words[i:i + number_of_words])

        # Split the words into multiple lines if they are too long
        wrapped_words = textwrap.wrap(segment_words, width=12)

        # Add the words to the plot
        for line_num, line in enumerate(wrapped_words):
            plt.text(middle_time, 0.95 - line_num * 0.1, line, horizontalalignment='center', verticalalignment='top')

    plt.title('Original Audio and Trim Points')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()