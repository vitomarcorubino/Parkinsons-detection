from collections import Counter
import explainability
from AudioClassifierCNN import AudioClassifier
import torch

model_filepath = "models/audio_classifierCNN.pth"

# Load the trained model
model = AudioClassifier()

model.load_state_dict(torch.load(model_filepath))

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
combined_not_parkinsons = dict(Counter(train_analysis[0]) + Counter(validation_analysis[0]) + Counter(test_analysis[0]))

# Combine the 'Parkinson's' analysis
combined_parkinsons = dict(Counter(train_analysis[1]) + Counter(validation_analysis[1]) + Counter(test_analysis[1]))

# Sort the 'Not Parkinson's' analysis in descending order by word frequency
combined_not_parkinsons = dict(sorted(combined_not_parkinsons.items(), key=lambda item: item[1], reverse=True))

# Sort the 'Parkinson's' analysis in descending order by word frequency
combined_parkinsons = dict(sorted(combined_parkinsons.items(), key=lambda item: item[1], reverse=True))

print("COMBINED LEXYCOMETRIC ANALYSIS")
print("Not Parkinson's: ", combined_not_parkinsons)
print("Parkinson's: ", combined_parkinsons)