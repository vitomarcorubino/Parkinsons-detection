import numpy as np
import matplotlib.pyplot as plt
import textwrap
import torch
from scipy.io.wavfile import read
import librosa
import librosa.display
from trimming import get_segment_times

def plot_heatmap(features, output, model, file_path, prediction):
    segment_times = get_segment_times(file_path)
    print(segment_times)

    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Create a time array to plot the seconds on the x-axis
    time = np.arange(0, len(audio)) / sample_rate

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Get the gradient of the output with respect to the parameters of the model
    output[:, predicted.item()].backward()

    # Pull the gradients out of the model
    gradients = model.get_activations_gradient()

    # Pool the gradients across the channels
    pooled_gradients = torch.mean(gradients, dim=[0, 2])

    # Get the activations of the last convolutional layer
    activations = model.get_activations(features).detach()

    # Weight the channels by corresponding gradients
    for i in range(96):  # 96 is the number of channels in the last conv layer
        activations[:, i, :] *= pooled_gradients[i]

    # Average the channels of the activations
    heatmap = torch.mean(activations, dim=1).squeeze()

    # Normalize the heatmap to make the values between 0 and 1
    heatmap = np.maximum(heatmap, 0)
    heatmap /= torch.max(heatmap)

    # Normalize the activations to the range of the audio signal
    if heatmap.dim() == 0:  # If heatmap is a 0-dimensional tensor (a scalar)
        normalized_activations = np.full(len(audio), heatmap.item())
    else:
        normalized_activations = np.interp(np.arange(len(audio)), np.linspace(0, len(audio), len(heatmap)), heatmap.numpy())

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
