import os
import numpy as np
import matplotlib.pyplot as plt
import textwrap
import torch
from scipy.io.wavfile import read
import librosa
import librosa.display
from trimming import get_segment_times, transcribe_audio
from explainability import get_normalized_activations

def plot_heatmap(file_path, model, features, output, prediction):
    segment_times = get_segment_times(file_path)
    # print("Segment times: ", segment_times)

    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Create a time array to plot the seconds on the x-axis
    time = np.arange(0, len(audio)) / sample_rate

    # Get the normalized activations
    normalized_activations = get_normalized_activations(file_path, model, features, output)

    # Calculate the 90th percentile of the normalized activations
    percentile_90 = np.percentile(normalized_activations, 90)


    # Create a list of tuples where each tuple contains the start and end indices of each segment
    segment_indices = [(int(start_time * sample_rate), int(end_time * sample_rate)) for start_time, end_time in
                       segment_times]

    # Create a list to store the indices of segment times that contain activations above the 90th percentile
    indices_with_activations_above_90 = []

    # Iterate over the segment indices
    for index, (start_index, end_index) in enumerate(segment_indices):
        # Get the activations for this segment
        segment_activations = normalized_activations[start_index:end_index]

        # Check if any of the activations in this segment are above the 90th percentile
        if any(segment_activations > percentile_90):
            # If there are any activations above the 90th percentile, store the index of the segment time
            indices_with_activations_above_90.append(index)

    print("Indices of segment times with activations above 90th percentile: ", indices_with_activations_above_90)

    # Iterate over indices with activations above the 90th percentile
    for i in indices_with_activations_above_90:
        # Get the directory and base file name without extension
        dir_name = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Create the path for the i-th trimmed segment
        segment_path = os.path.join(dir_name, "trimmed", f"{base_name}_trimmed{i}.wav")

        # Transcribe the audio segment
        transcription = transcribe_audio(segment_path)

        # Print the transcription
        print(f"Transcription of segment {i}: {transcription}")

    # Plot the waveform
    plt.figure(figsize=(10, 4))

    i = 0
    # Add vertical lines for each start and end time of the audio segments
    for start_time, end_time in segment_times:
        plt.axvline(x=start_time, color='0.8')  # Start of segment

        # Add a number for each segment at the top, between the vertical lines of start and end times
        segment_midpoint = (start_time + end_time) / 2
        plt.text(segment_midpoint, 0.75, str(i), horizontalalignment='center', verticalalignment='top', fontsize=8)

        plt.axvline(x=end_time, color='0.8')  # End of segment
        i = i + 1

    plt.plot(time, audio, alpha=1.0, label='Waveform')  # Use time array as x-values to display seconds

    # Check if time[-1] is 0
    if time[-1] == 0:
        time[-1] = 1e-10  # Set it to a small positive value

    # Display the heatmap as a 2D image
    plt.imshow(normalized_activations[np.newaxis, :], cmap='hot', aspect='auto', alpha=0.5, extent=(0.0, float(time[-1]), -1.0, 1.0))
    plt.colorbar(label='Activation Strength')  # Add label for the colorbar
    plt.xlabel('Time (s)')  # Add label for the x-axis
    plt.ylabel('Amplitude')  # Add label for the y-axis
    plt.legend()  # Add the legend


    # Add the file path and the prediction to the top left of the plot
    plt.text(-0.08, 1.135, f'File: {file_path}\nPrediction: {prediction}', horizontalalignment='left',
             verticalalignment='top', transform=plt.gca().transAxes)
    plt.show()  # Show the plot

def plot_time_frequency_heatmap(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path)

    # Compute the STFT of the audio signal
    d = librosa.stft(y)

    # Convert the amplitude of the STFT to decibels
    d_db = librosa.amplitude_to_db(abs(d), ref=np.max)

    # Create a new figure
    plt.figure(figsize=(12, 8))

    # Display the spectrogram as a heatmap
    librosa.display.specshow(d_db, sr=sr, x_axis='time', y_axis='log')

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
