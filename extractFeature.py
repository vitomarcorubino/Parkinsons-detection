import os
from featureExtraction import FeatureExtraction
import pickle

# Define the main dataset folder
main_folder = "datasetPeople_noVowels"

# Define the subfolders
subfolders = ["test", "validation", "train"]

# Create an instance of FeatureExtraction
featureExtraction = FeatureExtraction()

"""
with open('features/DL/train_features_transposed.pkl', 'rb') as file:
    features = pickle.load(file)
    features = np.array(features)
    print(features)


with open('features/DL/train_labels_transposed.pkl', 'rb') as file:
    labels = pickle.load(file)
    labels = np.array(labels)
    # print(labels)
"""
# Loop over the subfolders
for subfolder in subfolders:
    path = os.path.join(main_folder, subfolder)

    features, labels = featureExtraction.extract_features_from_folder4(path)  # for replicating the research paper

    # Serialize the features and labels
    with open(f"features/DL/people_noVowels/{subfolder}_features_transposed_split3.pkl", 'wb') as file:
        pickle.dump(features, file)

    with open(f"features/DL/people_noVowels/{subfolder}_labels_transposed_split3.pkl", 'wb') as file:
        pickle.dump(labels, file)