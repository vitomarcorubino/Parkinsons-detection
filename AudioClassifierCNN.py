import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import librosa  # for audio processing
import glob  # to retrieve files/pathnames matching a specified pattern
import matplotlib.pyplot as plt
from plotting import plot_heatmap
import pickle
from sklearn.model_selection import KFold
from featureExtraction import FeatureExtraction
import torch.nn.functional as F

class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = [torch.from_numpy(x) for x in X]
        self.y = [torch.tensor(label) for label in y]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        # Separate the inputs and labels
        inputs, labels = zip(*batch)

        # Find the maximum length of sequences among inputs
        max_length = max(input.shape[1] for input in inputs)

        # Pad the inputs to match the maximum length
        padded_inputs = []
        for input in inputs:
            padding_length = max_length - input.shape[1] # Calculate the padding length
            padding = torch.zeros((24, padding_length)) # Create a tensor of padding with -1 values
            padded_input = torch.cat((input, padding), dim=1) # Concatenate the input and padding
            padded_inputs.append(padded_input) # Append the padded input to the list

        # Stack the padded inputs
        inputs = torch.stack(padded_inputs)

        # Stack the labels
        labels = torch.stack(labels)

        return inputs, labels

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
    if (os.path.exists(f'pickledData/{trimmed}trainData.pkl') and os.path.exists(
            f'pickledData/{trimmed}valData.pkl') and
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

    return np.array(X_train), np.array(y_train), np.array(X_val), np.array(y_val), np.array(X_test), np.array(y_test)


# Define the model class
class AudioClassifier(nn.Module):
    def __init__(self):
        """
        This is the constructor for the AudioClassifier class. It initializes the layers of the neural network.

        The network architecture consists of two convolutional layers, each followed by a ReLU activation function
        and a max pooling layer. The output of the second max pooling layer is then flattened and passed through a
        fully connected layer. The output of the fully connected layer is then passed through a softmax function to
        get the final output probabilities for the two classes: Parkinson's and Not Parkinson's.

        The gradients of the activations of the second convolutional layer are also stored in order to implement
        explainability.

        Args:
            nn.Module: The base class for all neural network modules in PyTorch.
        """
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(24, 48, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer with a kernel size of 2
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer with a kernel size of 2
        self.fc = nn.Linear(96, 2)  # Fully connected layer

        # Store the gradients
        self.gradients = None

    # Hook for the gradients
    def activations_hook(self, grad):
        """
        This function is a hook that stores the gradients of the activations of the second convolutional layer.

        Args:
            grad: The gradients of the activations.
        """
        self.gradients = grad

    def forward(self, x):
        """
        This function defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network. It should have a shape of (batch_size, num_features).

        Returns:
            torch.Tensor: The output tensor after the forward pass.It contains the softmax probabilities for the two
                          classes: 'parkinson' and 'notParkinson'.
        """
        out = self.conv1(x)  # Pass the input through the first convolutional layer
        out = self.relu1(out)  # Apply the ReLU activation function

        # Check if the size of the input is less than the kernel size
        if out.shape[2] < self.maxpool1.kernel_size:
            # Pad the input to make it equal to the kernel size
            padding_size = self.maxpool1.kernel_size - out.shape[2]
            out = F.pad(out, (0, padding_size))

        out = self.maxpool1(out) # Apply max pooling in order to reduce the spatial dimensions of the output
        out = self.conv2(out)  # Pass the output through the second convolutional layer
        out = self.relu2(out)  # Apply the ReLU activation function

        out.requires_grad_(True)
        h = out.register_hook(self.activations_hook) # Register the hook

        if out.shape[2] < self.maxpool2.kernel_size:
            # Pad the input to make it equal to the kernel size
            padding_size = self.maxpool2.kernel_size - out.shape[2]
            out = F.pad(out, (0, padding_size))


        out = self.maxpool2(out) # Apply max pooling
        out = torch.mean(out, dim=2)
        out = self.fc(out)  # Pass the output through the fully connected layer
        out = torch.softmax(out, dim=1)  # Apply the softmax function to get the final output probabilities

        return out

    # Method for the gradient extraction
    def get_activations_gradient(self):
        """
        This function returns the gradients of the activations of the second convolutional layer.
        These gradients are stored during the forward pass, and are used later for explainability purposes,
        which means visualize the importance of different parts of the input in the decision made by the network.

        Returns:
            torch.Tensor: The gradients of the activations of the second convolutional layer.
        """
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
        """
        This function returns the activations of the second convolutional layer.
        These activations are used later for explainability purposes, which means visualize the importance of
        different parts of the input in the decision made by the network.

        Args:
            x (torch.Tensor): The input tensor to the network. It should have a shape of (batch_size, num_features).

        Returns:
            torch.Tensor: The activations of the second convolutional layer.
        """
        return self.relu2(self.conv2(self.maxpool1(self.relu1(self.conv1(x.unsqueeze(1))))))

def train_and_evaluate_model():
    lr = 0.0001
    n_epochs = 15
    batch_size = 48
    decay = 1e-4

    X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
    with open('features/splitted/noVowels/train_features_transposed.pkl', 'rb') as file:
        X_train = list(pickle.load(file).values())

        # Replace NaN values with 0 in X_train
        for i in range(len(X_train)):
            X_train[i] = np.nan_to_num(X_train[i])

    with open('features/splitted/noVowels/train_labels_transposed.pkl', 'rb') as file:
        y_train = pickle.load(file)

    with open('features/splitted/noVowels/validation_features_transposed.pkl', 'rb') as file:
        X_val = list(pickle.load(file).values())

    with open('features/splitted/noVowels/validation_labels_transposed.pkl', 'rb') as file:
        y_val = pickle.load(file)

    with open('features/splitted/noVowels/test_features_transposed.pkl', 'rb') as file:
        X_test = list(pickle.load(file).values())

    with open('features/splitted/noVowels/test_labels_transposed.pkl', 'rb') as file:
        y_test = pickle.load(file)

    # Create DataLoaders
    train_data = AudioDataset(X_train, y_train)
    val_data = AudioDataset(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=AudioDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True, collate_fn=AudioDataset.collate_fn)

    model = AudioClassifier()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()  # Cross Entropy loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)  # Adam optimizer

    model.train()  # Set the model to training mode

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(train_loader):  # Loop over the training data
            outputs = model(inputs.float())  # Forward pass
            labels = labels.long()  # Convert labels to long type
            loss = criterion(outputs, labels)  # Calculate the loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

        # Calculate validation loss and accuracy
        val_loss = 0
        correct_val = 0
        total_val = 0
        model.eval()
        with torch.no_grad():  # Disable gradient tracking for validation
            for inputs, labels in val_loader:  # Loop over the validation data
                outputs = model(inputs.float())  # Forward pass
                labels = labels.long()  # Convert labels to long type
                loss = criterion(outputs, labels)  # Calculate the loss
                val_loss = val_loss + loss.item()  # Accumulate the loss

                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader)  # Calculate the average validation loss
        val_accuracy = 100 * correct_val / total_val

        # Store loss values
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{n_epochs} Train Loss: {loss.item()} Val Loss: {val_loss} Val Accuracy: {val_accuracy}%')

    # Save the model
    torch.save(model.state_dict(), 'audio_classifier4.pth') # 4 for no vowels

    # Plot loss values
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def predict_audio(file_path, model):
    # Create an instance of FeatureExtraction
    feature_extraction = FeatureExtraction()

    # Extract features using the extract_features method
    features_dict = feature_extraction.extract_features(file_path)

    # Convert the features to a single numpy array and then to a PyTorch tensor
    features = torch.tensor(np.array(list(features_dict.values())))

    # Pass the features through the model to get the prediction
    output = model(features.float())

    # Get the predicted class
    _, predicted = torch.max(output.data, 1)

    # Convert the prediction to a readable format
    if predicted.item() == 0:
        prediction = "Not Parkinson's"
    else:
        prediction = "Parkinson's"

    return prediction