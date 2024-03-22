import torch
from AudioClassifier import predict_audio, AudioClassifier, train_and_evaluate_model
import glob
import os

# Load the trained model
model = AudioClassifier()

# Check if the model file exists
if not os.path.isfile('audio_classifier.pth'):
    # If the model file does not exist, train and evaluate the model
    train_and_evaluate_model()
else:
    # Load the model from the file
    model.load_state_dict(torch.load('audio_classifier.pth'))

model.eval()

# Use the function
directory_path = "trimmed"
# Get all .wav files in the directory
audio_files = glob.glob(directory_path + '/*.wav')

parkinsonCounter = 0
notParkinsonCounter = 0
# Iterate over the audio files and predict each one
for file_path in audio_files:
    prediction = predict_audio(file_path, model)
    print(f"The predicted class for the audio file {file_path} is: {prediction}")
    if prediction == "Parkinson's":
        parkinsonCounter += 1
    else:
        notParkinsonCounter += 1

print(f"Parkinson's: {parkinsonCounter}")
print(f"Not Parkinson's: {notParkinsonCounter}")