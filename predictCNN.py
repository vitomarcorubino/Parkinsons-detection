import torch
from AudioClassifierCNN import AudioClassifier, train_and_evaluate_model, predict_audio
import glob
import os

train = False  # Set the train flag to choose whether to train the model or not
predict = True

# Load the trained model
model = AudioClassifier()

# Check if the model file exists
if not os.path.isfile('audio_classifier4.pth'):
    # If the model file does not exist, train and evaluate the model
    train_and_evaluate_model()
else:
    # If the model file exist, load the model from the file if the train flag is set to False, otherwise train the model
    if train:
        train_and_evaluate_model()
    else:
        model.load_state_dict(torch.load('audio_classifier4.pth'))

if predict:
    # Set the model to evaluation mode
    model.eval()

    # Set the directory path of the audio files to predict
    # directory_path = "newDataset/elderlyHealthyControl/GiovannaAnaclerio/mono_pcm"
    directory_path = "newDataset/peopleWithParkinson/DonatoBruno/mono_pcm"
    # directory_path = "newDataset/elderlyHealthyControl/MariangelaColaianni/mono_pcm"
    # directory_path = "newDataset/youngHealthyControl/VitoMarcoRubino/mono_pcm"
    # directory_path = "dataset2/test/peopleWithParkinson/Mario B"
    # directory_path = "dataset2/test/elderlyHealthyControl/TERESA M"
    # Get all .wav files in the directory
    audio_files = glob.glob(directory_path + '/*.wav')

    parkinsonCounter = 0
    notParkinsonCounter = 0
    # Iterate over the audio files and predict each one
    for file_path in audio_files:
        prediction = predict_audio(file_path, model)
        print(f"The predicted class for the audio file {file_path} is: {prediction}")

        if prediction == "Parkinson's":
            parkinsonCounter = parkinsonCounter + 1
        else:
            notParkinsonCounter = notParkinsonCounter + 1

    if parkinsonCounter + notParkinsonCounter > 0:
        # Calculate the percentage of Parkinson's predictions
        parkinsonPercentage = (parkinsonCounter / (parkinsonCounter + notParkinsonCounter)) * 100
        print(f"The percentage of Parkinson's predictions is: {parkinsonPercentage:.2f}%")

    # Print the number of Parkinson's and not Parkinson's predictions
    print(f"Parkinson's: {parkinsonCounter}")
    print(f"Not Parkinson's: {notParkinsonCounter}")
