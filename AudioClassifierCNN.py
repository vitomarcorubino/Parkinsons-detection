import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from plotting import plot_heatmap
import pickle
from featureExtraction import FeatureExtraction
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix)


class AudioDataset(Dataset):
    def __init__(self, X, y):
        """
        Initializes the AudioDataset object.

        Args:
            X (list): A list of numpy arrays where each array represents the features of an audio sample.
                      These features are converted to PyTorch tensors and stored in self.X.

            y (list): A list of labels corresponding to each audio sample in X. Each label is converted to a
                      PyTorch tensor and stored in self.y.
        """
        self.X = [torch.from_numpy(x) for x in X]
        self.y = [torch.tensor(label) for label in y]

    def __len__(self):
        """
        Returns the total number of samples in the dataset.

        This method is used by PyTorch to know the number of samples in the dataset. It is used during the creation
        of a DataLoader object to determine the total number of batches.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns the features and label of the sample at the given index.

        This method is used by PyTorch to access the data of the dataset. It is used during the creation
        of a DataLoader object to fetch the samples.

        Args:
            idx (int): The index of the sample to fetch.

        Returns:
            tuple: A tuple containing the features and label of the sample at the given index.
        """
        return self.X[idx], self.y[idx]

    @staticmethod
    def collate_fn(batch):
        """
        This method is used to collate the samples into a batch. It separates the inputs and labels, finds the maximum
        length of sequences among inputs, pads the inputs to match the maximum length, and then stacks the padded inputs
        and labels.

        This method is used by PyTorch's DataLoader to collate the samples into a batch. It is necessary when the
        samples in a batch have varying sizes and need to be padded to the same size.

        Args:
            batch (list): A list of tuples where each tuple contains an input tensor and its corresponding label tensor.

        Returns:
            tuple: A tuple containing two tensors. The first tensor is a stack of the padded inputs and the second
            tensor is a stack of the labels.
        """
        # Separate the inputs and labels
        inputs, labels = zip(*batch)

        # Find the maximum length of sequences among inputs
        max_length = max(input.shape[1] for input in inputs)

        # Pad the inputs to match the maximum length
        padded_inputs = []
        for input in inputs:
            padding_length = max_length - input.shape[1]  # Calculate the padding length
            padding = torch.zeros((24, padding_length))  # Create a tensor of padding with -1 values
            padded_input = torch.cat((input, padding), dim=1)  # Concatenate the input and padding
            padded_inputs.append(padded_input)  # Append the padded input to the list

        # Stack the padded inputs
        inputs = torch.stack(padded_inputs)

        # Stack the labels
        labels = torch.stack(labels)

        return inputs, labels


# Define the model class
class AudioClassifier(nn.Module):
    def __init__(self):
        """
        Initializes the AudioClassifier model. This model is a subclass of the PyTorch nn.Module class.
        The model architecture consists of two convolutional layers, each followed by a ReLU activation function and a
        max pooling layer, and a final fully connected layer.

        The first convolutional layer has 24 input channels and 48 output channels, and uses a kernel size of 3 with
        padding of 1.
        The second convolutional layer has 48 input channels and 96 output channels, and also uses a kernel size of 3
        with padding of 1.
        Both max pooling layers use a kernel size of 2.
        The fully connected layer has 96 input features and 1 output feature.
        """
        super(AudioClassifier, self).__init__()
        self.conv1 = nn.Conv1d(24, 48, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()  # ReLU activation function
        self.maxpool1 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer with a kernel size of 3
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()  # ReLU activation function
        self.maxpool2 = nn.MaxPool1d(kernel_size=2)  # Max pooling layer with a kernel size of 3
        self.fc = nn.Linear(96, 1)  # Fully connected layer

        # Store the gradients
        self.gradients = None

    # Hook for the gradients
    def activations_hook(self, grad):
        """
        This hook is used to store the gradients of the activations of the second convolutional layer.

        Args:
            grad (torch.Tensor): The gradients of the activations of the second convolutional layer.
        """
        self.gradients = grad

    def forward(self, x):
        """
        This method defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor to the network. It should have a shape of (batch_size, num_features).

        Returns:
            torch.Tensor: The output tensor after passing through the network. It will have a shape of (batch_size, 1).

        """
        out = self.conv1(x)  # Pass the input through the first convolutional layer
        out = self.relu1(out)  # Apply the ReLU activation function

        # Check if the size of the input is less than the kernel size
        if out.shape[2] < self.maxpool1.kernel_size:
            # Pad the input to make it equal to the kernel size
            padding_size = self.maxpool1.kernel_size - out.shape[2]
            out = F.pad(out, (0, padding_size))

        out = self.maxpool1(out)  # Apply max pooling in order to reduce the spatial dimensions of the output
        out = self.conv2(out)  # Pass the output through the second convolutional layer
        out = self.relu2(out)  # Apply the ReLU activation function

        out.requires_grad_(True)
        h = out.register_hook(self.activations_hook)  # Register the hook

        if out.shape[2] < self.maxpool2.kernel_size:
            # Pad the input to make it equal to the kernel size
            padding_size = self.maxpool2.kernel_size - out.shape[2]
            out = F.pad(out, (0, padding_size))

        out = self.maxpool2(out)  # Apply max pooling
        out = torch.mean(out, dim=2)
        out = self.fc(out)  # Pass the output through the fully connected layer

        return out

    # Method for the gradient extraction
    def get_activations_gradient(self):
        """
        This function returns the gradients of the activations of the second convolutional layer.

        Returns:
            torch.Tensor: The gradients of the activations of the second convolutional layer.
        """
        return self.gradients

    def get_activations(self, x):
        """
        This function takes an input tensor and returns the activations of the second convolutional layer.

        Args:
            x (torch.Tensor): The input tensor to the network. It should have a shape of (batch_size, num_features).

        Returns:
            torch.Tensor: The activations of the second convolutional layer.
        """
        out = self.conv1(x)
        out = self.relu1(out)

        if out.shape[2] < self.maxpool1.kernel_size:
            padding_size = self.maxpool1.kernel_size - out.shape[2]
            out = F.pad(out, (0, padding_size))

        out = self.maxpool1(out)
        out = self.conv2(out)
        out = self.relu2(out)

        return out


def train_and_evaluate_model(model_filepath):
    """
    This function trains and evaluates the AudioClassifier model on the provided dataset.

    Args:
        model_filepath (str): The path to the file where the trained model will be saved.
    """
    lr = 0.00009
    n_epochs = 10
    batch_size = 32
    decay = 0.005

    X_train, y_train, X_val, y_val = [], [], [], []
    with open('features/DL/noVowels/train_features_transposed.pkl', 'rb') as file:
        X_train = list(pickle.load(file).values())

        # Replace NaN values with 0 in X_train
        for i in range(len(X_train)):
            X_train[i] = np.nan_to_num(X_train[i])

    with open('features/DL/noVowels/train_labels_transposed.pkl', 'rb') as file:
        y_train = pickle.load(file)

    with open('features/DL/noVowels/validation_features_transposed.pkl', 'rb') as file:
        X_val = list(pickle.load(file).values())

    with open('features/DL/noVowels/validation_labels_transposed.pkl', 'rb') as file:
        y_val = pickle.load(file)

    # Create DataLoaders
    train_data = AudioDataset(X_train, y_train)
    val_data = AudioDataset(X_val, y_val)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=AudioDataset.collate_fn)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                             collate_fn=AudioDataset.collate_fn)

    model = AudioClassifier()

    # Define loss and optimizer
    criterion = BCEWithLogitsLoss()  # Binary Cross Entropy loss with logits
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)  # Adam optimizer

    model.train()  # Set the model to training mode

    # Initialize lists to store loss values
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(n_epochs):
        for i, (inputs, labels) in enumerate(train_loader):  # Loop over the training data
            outputs = model(inputs.float())  # Forward pass
            labels = labels.float()  # Convert labels to float type
            loss = criterion(outputs, labels.unsqueeze(1))  # Calculate the loss

            optimizer.zero_grad()  # Zero the gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update the weights

        # Calculate validation loss and accuracy
        val_loss = 0
        model.eval()
        with torch.no_grad():  # Disable gradient tracking for validation
            for inputs, labels in val_loader:  # Loop over the validation data
                outputs = model(inputs.float())  # Forward pass
                labels = labels.float()  # Convert labels to float type
                loss = criterion(outputs, labels.unsqueeze(1))  # Calculate the loss
                val_loss = val_loss + loss.item()  # Accumulate the loss

        val_loss = val_loss / len(val_loader)  # Calculate the average validation loss

        # Store loss values
        train_losses.append(loss.item())
        val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{n_epochs} Train Loss: {loss.item()} Val Loss: {val_loss}')

    # Save the model
    torch.save(model.state_dict(), model_filepath)

    # Plot loss values
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def test_model(model):
    """
    This function tests the trained model on the test set and prints:
        - Accuracy
        - Precision
        - Recall
        - F1 Score
        - True values and predicted values
        - Classification Report
        - Confusion Matrix

    Args:
        model (AudioClassifier): The trained model.
    """
    model.eval()
    batch_size = 32
    X_test, y_test = [], []

    with open('features/DL/noVowels/test_features_transposed.pkl', 'rb') as file:
        X_test = list(pickle.load(file).values())

    with open('features/DL/noVowels/test_labels_transposed.pkl', 'rb') as file:
        y_test = pickle.load(file)

    test_loader = DataLoader(AudioDataset(X_test, y_test), batch_size=batch_size, shuffle=False,
                             collate_fn=AudioDataset.collate_fn)

    # Calculate metrics on test set
    y_pred = []
    with torch.no_grad():  # Disable gradient tracking for validation
        for inputs, labels in test_loader:  # Loop over the validation data
            outputs = model(inputs.float())  # Forward pass

            # Get predicted class
            predicted = torch.round(torch.sigmoid(outputs))
            y_pred.extend([int(p[0]) for p in predicted.tolist()])

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
    precision = round(precision_score(y_test, y_pred) * 100, 2)
    recall = round(recall_score(y_test, y_pred) * 100, 2)
    f1 = round(f1_score(y_test, y_pred) * 100, 2)

    # Store the metrics in a dictionary
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

    # Create a bar chart
    plt.bar(metrics.keys(), metrics.values())
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Metrics')
    plt.ylim([0, 100])  # Set the limit of y-axis to match the percentage
    plt.show()

    mismatches = []

    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            mismatches.append(i)

    # Print Y_test size and y_test values
    print(f"Y_test ({len(y_test)}): " + str(y_test))
    print(f"Y_pred ({len(y_pred)}): " + str(y_pred))
    print(f"Mismatches ({len(mismatches)}):" + str(mismatches))
    print("------------------------------------")
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}%')
    print(f'Recall: {recall}%')
    print(f'F1 Score: {f1}%')

    target_names = ["Not Parkinson's", "Parkinson's"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    conf_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Confusion matrix\n", conf_matrix)

    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

import pandas as pd

def predict_audio(file_path, model, heatmap):
    """
    This function takes an audio file path, a trained model, and a boolean value indicating whether to generate a heatmap,
    and returns the predicted class for the audio file.

    Args:
        file_path (str): The path to the audio file.
        model (AudioClassifier): The trained model.
        heatmap (bool): A boolean value indicating whether to generate a heatmap for explainability purposes.

    Returns:
        str: The predicted class for the audio file.
    """
    # Create an instance of FeatureExtraction
    feature_extraction = FeatureExtraction()

    # Extract features using the extract_features method
    features_dict = feature_extraction.extract_features(file_path)

    # Convert the features to a single numpy array and then to a PyTorch tensor
    features = torch.tensor(np.array(list(features_dict.values())))

    # Pass the features through the model to get the prediction
    output = model(features.float())

    predicted = torch.round(torch.sigmoid(output))

    if predicted == 0:
        prediction = "Not Parkinson's"
    else:
        prediction = "Parkinson's"

    if heatmap:
        plot_heatmap(file_path, model, features.float(), output, prediction)

    return prediction

