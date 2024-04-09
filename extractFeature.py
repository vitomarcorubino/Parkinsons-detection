import os
from featureExtraction import FeatureExtraction
import pandas as pd

extract_trimmed = False  # Set to True to extract features from the trimmed dataset

trimmed_subfolder = ""
trimmed_filename = ""
if extract_trimmed:
    trimmed_filename = "_trimmed"
    trimmed_subfolder = "trimmed/"


# Define the main dataset folder
main_folder = "dataset2"

# Define the subfolders
subfolders = ["train", "validation", "test"]

# Define the labels
labels = ["elderlyHealthyControl", "peopleWithParkinson", "youngHealthyControl"]

# Create an instance of FeatureExtraction
f = FeatureExtraction()

# Loop over the subfolders
for subfolder in subfolders:
    # Initialize a list to store the features
    features_list = []

    # Loop over the labels
    for label in labels:
        # Define the path for the current label
        path = os.path.join(main_folder, subfolder, label, f"*/{trimmed_subfolder}*.wav")
        print(path)

        features = f.extract_features_from_folder(path)  # for replicating the research paper

        if label == "peopleWithParkinson":
            features['label'] = 1
        else:
            features['label'] = 0
        features_list.append(features)

    # Concatenate all the DataFrames in the list into a single DataFrame
    features_df = pd.concat(features_list)

    f.convert_to_csv(features_df, f"features/splitted/{trimmed_subfolder}{subfolder}_features{trimmed_filename}")

"""
import os
from featureExtraction import FeatureExtraction
import pandas as pd

extract_trimmed = False  # Set to True to extract features from the trimmed dataset

trimmed_subfolder = ""
trimmed_filename = ""
if extract_trimmed:
    trimmed_filename = "_trimmed"
    trimmed_subfolder = "trimmed/"

# Define the main dataset folder
main_folder = "datasetVowelsShuffled"

# Define the labels
labels = ["elderlyHealthyControl", "peopleWithParkinson"]

# Create an instance of FeatureExtraction
f = FeatureExtraction()

# Initialize a list to store the features
features_list = []

# Loop over the labels
for label in labels:
    # Define the path for the current label
    path = os.path.join(main_folder, label, f"{trimmed_subfolder}*.wav")
    print(path)

    features = f.extract_features_from_folder(path)  # for replicating the research paper

    if label == "peopleWithParkinson":
        features['label'] = 1
    else:
        features['label'] = 0
    features_list.append(features)

# Concatenate all the DataFrames in the list into a single DataFrame
features_df = pd.concat(features_list)

f.convert_to_csv(features_df, f"features/shuffledVowels/{trimmed_subfolder}shuffledVowels_features{trimmed_filename}")
"""