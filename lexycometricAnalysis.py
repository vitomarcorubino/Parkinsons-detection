from collections import Counter
import explainability
from AudioClassifierCNN import AudioClassifier
import torch
import pickle
import os

model_filepath = "models/audio_classifierCNN.pth"
not_parkinson_filepath = "crucialWords/notParkinsonsCNN.pkl"
parkinson_filepath = "crucialWords/parkinsonCNN.pkl"

# Load the trained model
model = AudioClassifier()

model.load_state_dict(torch.load(model_filepath))

if not os.path.exists(not_parkinson_filepath) and not os.path.exists(parkinson_filepath):
    train_analysis = explainability.lexycometric_analysis_on_set("dataset/train", model)
    validation_analysis = explainability.lexycometric_analysis_on_set("dataset/validation", model)
    test_analysis = explainability.lexycometric_analysis_on_set("dataset/test", model)

    print("TRAIN SET LEXYCOMETRIC ANALYSIS")
    print("Not Parkinson's: ", train_analysis[0])
    print("Parkinson's: ", train_analysis[1])

    print("VALIDATION SET LEXYCOMETRIC ANALYSIS")
    print("Not Parkinson's: ", validation_analysis[0])
    print("Parkinson's: ", validation_analysis[1])

    print("TEST SET LEXYCOMETRIC ANALYSIS")
    print("Not Parkinson's: ", test_analysis[0])
    print("Parkinson's: ", test_analysis[1])

    # Combine the 'Not Parkinson's' analysis
    combined_not_parkinsons = dict(
        Counter(train_analysis[0]) + Counter(validation_analysis[0]) + Counter(test_analysis[0]))

    # Combine the 'Parkinson's' analysis
    combined_parkinsons = dict(Counter(train_analysis[1]) + Counter(validation_analysis[1]) + Counter(test_analysis[1]))

    # Sort the 'Not Parkinson's' analysis in descending order by word frequency
    combined_not_parkinsons = dict(sorted(combined_not_parkinsons.items(), key=lambda item: item[1], reverse=True))

    # Sort the 'Parkinson's' analysis in descending order by word frequency
    combined_parkinsons = dict(sorted(combined_parkinsons.items(), key=lambda item: item[1], reverse=True))

    # Save the combined analysis to a pickle file
    with open(not_parkinson_filepath, 'wb') as f:
        pickle.dump(combined_not_parkinsons, f)

    with open(parkinson_filepath, 'wb') as f:
        pickle.dump(combined_parkinsons, f)
else:
    # Open the pickle file
    with open(not_parkinson_filepath, 'rb') as f:
        combined_not_parkinsons = pickle.load(f)

    with open(parkinson_filepath, 'rb') as f:
        combined_parkinsons = pickle.load(f)

print("COMBINED LEXYCOMETRIC ANALYSIS")
print("Not Parkinson's: ", combined_not_parkinsons)
print("Parkinson's: ", combined_parkinsons)

# Convert the keys of both dictionaries to sets
not_parkinsons_words = set(combined_not_parkinsons.keys())
parkinsons_words = set(combined_parkinsons.keys())

# Find the common words
common_words = not_parkinsons_words.intersection(parkinsons_words)

# Print the common words
print("\nCommon words:", common_words)

# Find the unique words in each dictionary
unique_not_parkinsons_words = not_parkinsons_words.difference(parkinsons_words)
unique_parkinsons_words = parkinsons_words.difference(not_parkinsons_words)

# Create new dictionaries with only the unique words
unique_not_parkinsons = {word: combined_not_parkinsons[word] for word in unique_not_parkinsons_words}
unique_parkinsons = {word: combined_parkinsons[word] for word in unique_parkinsons_words}

# Sort unique words in descending order by word frequency
unique_not_parkinsons = dict(sorted(unique_not_parkinsons.items(), key=lambda item: item[1], reverse=True))
unique_parkinsons = dict(sorted(unique_parkinsons.items(), key=lambda item: item[1], reverse=True))

# Print the unique words
print("\nUnique words in 'Not Parkinson's':", unique_not_parkinsons)
print("Unique words in 'Parkinson's':", unique_parkinsons)