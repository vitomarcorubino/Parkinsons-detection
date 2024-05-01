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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, classification_report,
                             confusion_matrix)


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
        self.fc = nn.Linear(96, 1)  # Fully connected layer

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
        These gradients are stored during the forward pass, and are used later for explainability purposes,
        which means visualize the importance of different parts of the input in the decision made by the network.

        Returns:
            torch.Tensor: The gradients of the activations of the second convolutional layer.
        """
        return self.gradients

    # Method for the activation extraction
    def get_activations(self, x):
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
    lr = 0.0001
    n_epochs = 20
    batch_size = 48
    decay = 0.005

    X_train, y_train, X_val, y_val = [], [], [], []
    with open('features/DL/train_features_transposed.pkl', 'rb') as file:
        X_train = list(pickle.load(file).values())

        # Replace NaN values with 0 in X_train
        for i in range(len(X_train)):
            X_train[i] = np.nan_to_num(X_train[i])

    with open('features/DL/train_labels_transposed.pkl', 'rb') as file:
        y_train = pickle.load(file)

    with open('features/DL/validation_features_transposed.pkl', 'rb') as file:
        X_val = list(pickle.load(file).values())

    with open('features/DL/validation_labels_transposed.pkl', 'rb') as file:
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
    model.eval()
    batch_size = 48
    X_test, y_test = [], []

    with open('features/DL/test_features_transposed.pkl', 'rb') as file:
        X_test = list(pickle.load(file).values())

    with open('features/DL/test_labels_transposed.pkl', 'rb') as file:
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

    print("Y_test: ", y_test)
    print("Y_pred: ", y_pred)

    mismatches = []

    for i in range(len(y_pred)):
        if y_pred[i] != y_test[i]:
            mismatches.append(i)

    print("Indexes where values don't correspond:", mismatches)
    print("------------------------------------")
    print(f'Accuracy: {accuracy}%')
    print(f'Precision: {precision}%')
    print(f'Recall: {recall}%')
    print(f'F1 Score: {f1}%')

    target_names = ["Not Parkinson's", "Parkinson's"]
    print(classification_report(y_test, y_pred, target_names=target_names))

    print("Confusion matrix\n", confusion_matrix(y_test, y_pred, labels=[0, 1]))


def predict_audio(file_path, model, heatmap):
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
