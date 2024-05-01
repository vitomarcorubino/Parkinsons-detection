import os
import glob
import numpy as np
import torch
import librosa
import librosa.display
from featureExtraction import FeatureExtraction
from trimming import get_segment_times, transcribe_audio
from collections import Counter


def get_normalized_activations(file_path, model, features, output):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Get the gradient of the output with respect to the parameters of the model
    output[:, predicted.item()].backward(retain_graph=True)

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
        normalized_activations = np.interp(np.arange(len(audio)), np.linspace(0, len(audio), len(heatmap)),
                                           heatmap.numpy())

    return normalized_activations


def get_segment_indices_with_activations_above_90(file_path, model, features, output):
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    segment_times = get_segment_times(file_path)

    normalized_activations = get_normalized_activations(file_path, model, features.float(), output)

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

    return indices_with_activations_above_90


def lexycometric_analysis_on_folder(folder_path, model):
    feature_extraction = FeatureExtraction()

    # Get all .wav files in the directory
    audio_files = glob.glob(folder_path + '/*.wav')

    word_frequencies = {}

    # Iterate over the audio files and predict each one
    for file_path in audio_files:

        # Extract features using the extract_features method
        features_dict = feature_extraction.extract_features(file_path)

        # Convert the features to a single numpy array and then to a PyTorch tensor
        features = torch.tensor(np.array(list(features_dict.values())))

        if (features.size(2) > 2):  # If the audio has more than 2 segments
            print("Processing: ", file_path)

            # Pass the features through the model to get the prediction
            output = model(features.float())

            indices_with_activations_above_90 = get_segment_indices_with_activations_above_90(file_path, model,
                                                                                              features.float(), output)

            print("Indices with activations above 90th percentile: ", indices_with_activations_above_90)

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

                # Split the transcription into words and update their count in the dictionary
                for word in transcription.split():
                    if word in word_frequencies:
                        word_frequencies[word] = word_frequencies[word] + 1
                    else:
                        word_frequencies[word] = 1

    # Return the word frequencies in ascending order of count
    sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda item: item[1], reverse=True)

    return sorted_word_frequencies


def lexycometric_analysis_on_set(set_path, model):
    model.eval()

    # Define the folders
    folders = ["elderlyHealthyControl", "youngHealthyControl", "peopleWithParkinson"]

    # Initialize a list to store the word frequencies for each class
    word_frequencies_by_class = []

    # Initialize a Counter object to store the combined word frequencies of the first two folders
    combined_word_frequencies = Counter()

    # Iterate over the first two folders
    for folder in folders[:2]:
        # Construct the full path to the folder
        folder_path = os.path.join(set_path, folder)

        # Call the lexycometric_analysis_on_folder function and get the word frequencies
        word_frequencies = dict(lexycometric_analysis_on_folder(folder_path, model))

        # Add the word frequencies to the combined_word_frequencies Counter
        combined_word_frequencies += Counter(word_frequencies)

    # Convert the Counter to a dictionary and add it to the list
    combined_word_frequencies_dict = dict(combined_word_frequencies)

    # Sort the dictionary based on the count of words
    sorted_combined_word_frequencies = sorted(combined_word_frequencies_dict.items(), key=lambda item: item[1],
                                              reverse=True)

    # Convert the sorted list of tuples back to a dictionary
    sorted_combined_word_frequencies_dict = dict(sorted_combined_word_frequencies)

    # Add the sorted dictionary to the list
    word_frequencies_by_class.append(sorted_combined_word_frequencies_dict)

    # Construct the full path to the third folder
    folder_path = os.path.join(set_path, folders[2])

    # Call the lexycometric_analysis_on_folder function and get the word frequencies
    word_frequencies = dict(lexycometric_analysis_on_folder(folder_path, model))

    # Add the word frequencies of the third folder to the list
    word_frequencies_by_class.append(word_frequencies)

    # Return the word frequencies by class
    return word_frequencies_by_class

