import torch
from AudioClassifierCNN import AudioClassifier, train_and_evaluate_model, predict_audio, test_model
import glob
import os
from collections import Counter
from explainability import lexycometric_analysis_on_set

train = False  # Set the train flag to choose whether to train the model or not
predict = False
heatmap = False
test = False

model_filepath = "models/audio_classifierCNN.pth"

# Load the trained model
model = AudioClassifier()

# Check if the model file exists
if not os.path.isfile(model_filepath):
    # If the model file does not exist, train and evaluate the model
    train_and_evaluate_model(model_filepath)
else:
    # If the model file exist, load the model from the file if the train flag is set to False, otherwise train the model
    if train:
        train_and_evaluate_model(model_filepath)
    model.load_state_dict(torch.load(model_filepath))

train_analysis = lexycometric_analysis_on_set("dataset/train", model)
validation_analysis = lexycometric_analysis_on_set("dataset/validation", model)
test_analysis = lexycometric_analysis_on_set("dataset/test", model)

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


if test:
    test_model(model)

if predict:
    # Set the model to evaluation mode
    model.eval()

    # Set the directory path of the audio files to predict
    # directory_path = "datasetNew/elderlyHealthyControl/GiovannaAnaclerio/mono_pcm"
    # directory_path = "datasetNew/peopleWithParkinson/DonatoBruno/mono_pcm"
    # directory_path = "datasetNew/elderlyHealthyControl/MariangelaColaianni/mono_pcm"
    # directory_path = "datasetNew/youngHealthyControl/VitoMarcoRubino/mono_pcm"
    directory_path = "datasetPeople/test/peopleWithParkinson/Mario B"
    # directory_path = "datasetPeople/test/elderlyHealthyControl/TERESA M"
    # Get all .wav files in the directory
    audio_files = glob.glob(directory_path + '/*.wav')

    parkinsonCounter = 0
    notParkinsonCounter = 0
    # Iterate over the audio files and predict each one
    for file_path in audio_files:
        prediction = predict_audio(file_path, model, heatmap)
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