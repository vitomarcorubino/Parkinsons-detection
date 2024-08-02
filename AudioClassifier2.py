import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa  # for audio processing
import glob  # to retrieve files/pathnames matching a specified pattern
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold # for cross-validation
import pickle


class AudioDataset(Dataset):
    """
    Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, X, y):
        """
        This function initializes the dataset with the features and labels.

        Args:
            X (np.array): The features.
            y (np.array): The labels.
        """
        self.X = X
        self.y = y
        self.label_map = {"parkinson": 0, "notParkinson": 1}

    def __len__(self):
        """
        This function returns the size of the dataset.

        Returns:
            int: The size of the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        This function returns a sample from the dataset at the specified index.

        Args:
            idx (int): The index of the sample.
        """
        # Convert labels to numerical values and then to tensor
        labels = torch.tensor(self.label_map[self.y[idx]])
        return self.X[idx], labels



def pickle_dataset(X_train, y_train, X_test, y_test, X_val, y_val, pickle_trimmed):
    trimmed = ""
    if pickle_trimmed:
        trimmed = "trimmed/"

    # Create a dictionary to store X_train and y_train
    train_data = {'X_train': X_train, 'y_train': y_train}

    # Pickle the dictionary into a single file
    with open(f'pickledData/{trimmed}trainData.pkl', 'wb') as f:
        pickle.dump(train_data, f)

    # Create a dictionary to store X_val and y_val
    val_data = {'X_val': X_val, 'y_val': y_val}

    # Pickle the dictionary into a single file
    with open(f'pickledData/{trimmed}valData.pkl', 'wb') as f:
        pickle.dump(val_data, f)

    # Create a dictionary to store X_test and y_test
    test_data = {'X_test': X_test, 'y_test': y_test}

    # Pickle the dictionary into a single file
    with open(f'pickledData/{trimmed}testData.pkl', 'wb') as f:
        pickle.dump(test_data, f)


def load_data(load_trimmed):
    """
    This function loads the audio data and extracts the features using librosa.

    Returns:
        tuple: The features and labels for training, testing and validation sets.
    """

    trimmed = ""
    if load_trimmed:
        trimmed = "trimmed/"

    # Initialize the features and labels for each set
    X_train, y_train, X_test, y_test, X_val, y_val = [], [], [], [], [], []

    # Check if pickle files already exist
    if (os.path.exists(f'pickledData/{trimmed}trainData.pkl') and os.path.exists(f'pickledData/{trimmed}valData.pkl') and
            os.path.exists(f'pickledData/{trimmed}testData.pkl')):
        # Load the pickle files
        with open(f'pickledData/{trimmed}trainData.pkl', 'rb') as file:
            train_data = pickle.load(file)
            X_train, y_train = train_data['X_train'], train_data['y_train']

        with open(f'pickledData/{trimmed}valData.pkl', 'rb') as file:
            val_data = pickle.load(file)
            X_val, y_val = val_data['X_val'], val_data['y_val']

        with open(f'pickledData/{trimmed}testData.pkl', 'rb') as file:
            test_data = pickle.load(file)
            X_test, y_test = test_data['X_test'], test_data['y_test']
    else:
        # If pickle files do not exist, extract features using librosa
        # Define the sets
        sets = ['train', 'test', 'validation']
        """
        Define the category names of each set, where elderlyHealthyControl and youngHealthyControl are considered as 
        notParkinson
        """
        categories = ['elderlyHealthyControl', 'peopleWithParkinson', 'youngHealthyControl']

        # Define the path to the dataset
        data = {set_name: {category: f'dataset2/{set_name}/{category}' for category in categories} for set_name in sets}

        for set_name, set_data in data.items():  # Iterate over the sets
            for category, dir in set_data.items():  # Iterate over the categories
                # Get all subfolders in the specified directory
                subfolders = [f.path for f in os.scandir(dir) if f.is_dir()]

                for subfolder in subfolders:
                    # Get all .wav files in the subfolder
                    audio_files = glob.glob(subfolder + '/trimmed/*.wav')

                    for file in audio_files:
                        # Extract features using librosa
                        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')  # resample to 22050 Hz
                        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # extract 40 MFCCs
                        mfccs_processed = np.mean(mfccs.T, axis=0)  # average the MFCCs across all the frames

                        # Add the features to the appropriate list
                        if category == "peopleWithParkinson":
                            label = "parkinson"
                        else:
                            label = "notParkinson"

                        if set_name == 'train':
                            X_train.append(mfccs_processed)
                            y_train.append(label)
                        else:
                            if set_name == 'test':
                                X_test.append(mfccs_processed)
                                y_test.append(label)
                            else:  # validation set
                                X_val.append(mfccs_processed)
                                y_val.append(label)

        # Call the pickle_dataset function
        pickle_dataset(np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_val),
                       np.array(y_val), load_trimmed)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test), np.array(X_val), np.array(y_val)

# Define the model
class AudioClassifier(nn.Module):
    def __init__(self):
        """
        This function initializes the audio classifier model.

        The model consists of an LSTM layer followed by a hidden layer and an output layer.
        The LSTM layer has an input size of 40, hidden size of 128, 2 layers, and a dropout of 0.5.
        The hidden layer has an input size of 128 and an output size of 64 and uses the ReLU activation function.
        The output layer has an input size of 64 and an output size of 2 and uses the softmax activation function to
        output the class probabilities.
        """
        super(AudioClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=40, hidden_size=128, num_layers=2, dropout=0.5, batch_first=True)
        self.hidden_layer = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(64, 2)

    def forward(self, x):
        """
        This function defines the forward pass of the model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after the forward pass. It contains the softmax probabilities for the two
            classes: 'parkinson' and 'notParkinson'.
        """
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out.view(out.shape[0], -1)
        out = self.hidden_layer(out)
        out = self.relu(out)
        out = self.output_layer(out)
        out = torch.softmax(out, dim=1)
        return out


def train_and_evaluate_model(train_on_trimmed):
    """
    This function trains and evaluates the audio classifier model, using:
    - 3-fold cross-validation
    - Adam optimizer with a learning rate of 0.00003
    - Cross-entropy loss function
    - 10 epochs
    - Batch size of 48

    Finally, it saves the trained model as 'audio_classifier2.pth' and plots the training and validation loss values.
    """

    # Load the data: X for features, y for labels
    X_train, y_train, X_test, y_test, X_val, y_val = load_data(train_on_trimmed)

    # Create DataLoaders
    train_data = AudioDataset(X_train, y_train)
    test_data = AudioDataset(X_test, y_test)
    val_data = AudioDataset(X_val, y_val)

    # Define the number of folds
    n_folds = 3
    kfold = KFold(n_splits=n_folds, shuffle=True)

    model = AudioClassifier()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross-entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Adam optimizer with learning rate of 0.00003

    model.train()  # Set the model to training mode
    n = 30  # Number of epochs
    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(n):  # Iterate over the epochs

        """
        for fold, (train_ids, val_ids) in enumerate(kfold.split(train_data)):  # Iterate over the folds
            
            print(f'FOLD {fold + 1}')
            # Sample elements randomly from a given list of ids, no replacement.
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)

            # Define data loaders for training and testing data in this fold
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=24, sampler=train_subsampler)
            val_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=24, sampler=val_subsampler)
            """

        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=24)
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=24)
        loss = None
        for i, (inputs, labels) in enumerate(train_loader):  # Iterate over the training data
            inputs = inputs.view(inputs.size(0), -1)  # Reshape the input data
            outputs = model(inputs.float())  # Forward pass
            labels = labels.long()  # Convert labels to long
            loss = criterion(outputs, labels)  # Calculate the loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

        # Calculate validation loss
        val_loss = 0
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient tracking
            for inputs, labels in val_loader:  # Iterate over the validation data
                inputs = inputs.view(inputs.size(0), -1)  # Reshape the input data
                outputs = model(inputs.float())  # Forward pass
                labels = labels.long()  # Convert labels to long
                loss = criterion(outputs, labels)  # Calculate the loss
                val_loss = val_loss + loss.item()

        val_loss = val_loss / len(val_loader)  # Calculate the average validation loss

        # Store loss values
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{n} Train Loss: {loss.item()} Val Loss: {val_loss}')

    # Save the model
    torch.save(model.state_dict(), 'models/audio_classifier2.pth')

    model.eval()  # Set the model to evaluation mode

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=24  # Adjust the batch size according to your needs
    )

    # Evaluation
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient tracking
        for inputs, labels in test_loader:  # Iterate over the test data
            inputs = inputs.view(inputs.size(0), -1)  # Reshape the input data
            outputs = model(inputs.float())  # Forward pass
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class
            total = total + labels.size(0)  # Update the total count
            correct = correct + torch.sum(predicted.eq(labels)).item()  # Update the correct count

    print(f'Accuracy: {100 * correct / total}%')  # Print the accuracy

    # Plot loss values
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def predict_audio(file_path, model):
    """
    This function predicts the class of an audio file using the trained model.

    Args:
        file_path (str): The path to the audio file that needs to be classified.
        model (AudioClassifier): The trained model.
    """
    # Load the audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_processed = np.mean(mfccs.T, axis=0)

    # Convert the MFCCs to PyTorch tensor
    mfccs_tensor = torch.tensor(mfccs_processed).float().unsqueeze(0)

    # Pass the tensor through the model
    output = model(mfccs_tensor)

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Interpret the prediction
    if predicted.item() == 0:
        return "Parkinson's"
    else:
        return "Not Parkinson's"

'''
# Define the main dataset folder
main_folder = "dataset"

# Define the subfolders
subfolders = ["train", "test", "validation"]

# Define the labels
labels = ["elderlyHealthyControl", "peopleWithParkinson", "youngHealthyControl"]

# Initialize a list to store lengths of all audio files
lengths = []

# Loop over the subfolders
for subfolder in subfolders:
    # Loop over the labels
    for label in labels:
        # Define the path for the current label
        path = os.path.join(main_folder, subfolder, label, "*.wav")
        # Get all audio files in the current path
        audio_files = glob.glob(path)
        # Loop over the audio files
        for file_path in audio_files:
            # Load the audio file
            audio, sample_rate = librosa.load(file_path, sr=None)
            # Calculate the duration in seconds and append to the list
            print(f"Duration of {file_path}: {librosa.get_duration(y=audio, sr=sample_rate)} seconds")
            lengths.append(librosa.get_duration(y=audio, sr=sample_rate))

# Calculate the average length
average_length = sum(lengths) / len(lengths) if lengths else 0

print(f"Average length of audio files: {average_length} seconds")
print("Number of audio files: ", len(lengths))
'''
