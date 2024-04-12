import os
from featureExtraction import FeatureExtraction
import pandas as pd
import pickle
import numpy as np

# Define the main dataset folder
main_folder = "dataset3"

# Define the subfolders
subfolders = ["train", "validation", "test"]

# Create an instance of FeatureExtraction
featureExtraction = FeatureExtraction()


with open('features/splitted/test_features.pkl', 'rb') as file:
    features = pickle.load(file)


with open('features/splitted/test_labels.pkl', 'rb') as file:
    labels = pickle.load(file)
"""
# Loop over the subfolders
for subfolder in subfolders:
    path = os.path.join(main_folder, subfolder)

    features, labels = featureExtraction.extract_features_from_folder(path)  # for replicating the research paper

    # Serialize the features and labels
    with open(f"features/splitted/{subfolder}_features.pkl", 'wb') as file:
        pickle.dump(features, file)

    with open(f"features/splitted/{subfolder}_labels.pkl", 'wb') as file:
        pickle.dump(labels, file)

    # Convert the list of dictionaries to a DataFrame
    # features_df = pd.DataFrame(features)

    # f.convert_to_csv(features_df, f"features/splitted/{subfolder}_features")
"""