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

"""
with open('features/splitted/train_features.pkl', 'rb') as file:
    features = pickle.load(file)
    features = np.array(features)
    print(features)


with open('features/splitted/train_labels.pkl', 'rb') as file:
    labels = pickle.load(file)
    labels = np.array(labels)
    # print(labels)
"""
# Loop over the subfolders
for subfolder in subfolders:
    path = os.path.join(main_folder, subfolder)

    features, labels = featureExtraction.extract_features_from_folder(path)  # for replicating the research paper

    # Serialize the features and labels
    with open(f"features/splitted/{subfolder}_features_NT.pkl", 'wb') as file:
        pickle.dump(features, file)

    with open(f"features/splitted/{subfolder}_labels_NT.pkl", 'wb') as file:
        pickle.dump(labels, file)

    # Convert the list of dictionaries to a DataFrame
    # features_df = pd.DataFrame(features)

    # f.convert_to_csv(features_df, f"features/splitted/{subfolder}_features")