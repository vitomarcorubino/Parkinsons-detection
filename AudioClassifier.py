import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import librosa  # for audio processing
import glob  # to retrieve files/pathnames matching a specified pattern


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
        return self.X[idx], self.y[idx]


def load_data():
    """
    This function loads the audio data and extracts the features using librosa.

    Returns:
        tuple: The features and labels.
    """
    young_healthy_dir = 'dataset/youngHealthyControl/trimmed/*.wav'
    elderly_healthy_dir = 'dataset/elderlyHealthyControl/trimmed/*.wav'
    parkinsons_dir = 'dataset/peopleWithParkinson/trimmed/*.wav'

    # Get all .wav files in the specified directories
    young_healthy_files = glob.glob(young_healthy_dir)
    elderly_healthy_files = glob.glob(elderly_healthy_dir)
    parkinsons_files = glob.glob(parkinsons_dir)

    # Combine all files into one list
    audio_files = young_healthy_files + elderly_healthy_files + parkinsons_files

    X = []  # features
    y = []  # labels

    for file in audio_files:
        # Extract features using librosa
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')  # resample to 22050 Hz
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)  # extract 40 MFCCs
        mfccs_processed = np.mean(mfccs.T, axis=0)  # average the MFCCs across all the frames

        X.append(mfccs_processed)  # Add the features to the list

        # Add labels based on the audio file
        """
        if file in young_healthy_files:
            y.append(0)
        else:
            if file in elderly_healthy_files:
                y.append(1)
            else:
                if file in parkinsons_files:
                    y.append(2)
        """

        if file in parkinsons_files:
            y.append(0)
        else:
            y.append(1)

    return np.array(X), np.array(y)


# Define the model
class AudioClassifier(nn.Module):
    """
    This class defines the audio classifier model.

    Args:
        nn.Module: The PyTorch module class.
    """

    def __init__(self):
        """
        This function initializes the model layers.

        Args:
            self: The object pointer.
        """
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Linear(40, 128)
        self.dropout1 = nn.Dropout(0.7)
        self.layer2 = nn.Linear(128, 128)
        self.dropout2 = nn.Dropout(0.7)
        self.layer3 = nn.Linear(128, 64)
        self.dropout3 = nn.Dropout(0.7)
        self.layer4 = nn.Linear(64, 3)

    def forward(self, x):
        """
        This function defines the forward pass of the model.

        Args:
            self: The object pointer.
            x: The input features.
        """
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout2(x)
        x = torch.relu(self.layer3(x))
        x = self.dropout3(x)
        x = torch.softmax(self.layer4(x), dim=1)
        return x


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
    """
    if predicted.item() == 0:
        return "Young Healthy"
    else:
        if predicted.item() == 1:
            return "Elderly Healthy"
        else:
            if predicted.item() == 2:
                return "Parkinson's"
    """

    if predicted.item() == 0:
        return "Parkinson's"
    else:
        return "Not Parkinson's"


# Training and evaluation code
def train_and_evaluate_model():
    X, y = load_data()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Create DataLoaders
    train_data = AudioDataset(X_train, y_train)
    test_data = AudioDataset(X_test, y_test)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=True)

    model = AudioClassifier()

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.1)

    model.train()

    n = 50
    # Training loop
    for epoch in range(n):  # Increase the number of epochs
        loss = None
        for i, (inputs, labels) in enumerate(train_loader):
            outputs = model(inputs.float())
            labels = labels.long()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss is not None:
            print(f'Epoch {epoch + 1}/{n} Loss: {loss.item()}')
        else:
            print(f'Epoch {epoch + 1}/{n} No loss calculated')

    # Save the model
    torch.save(model.state_dict(), 'audio_classifier.pth')

    model.eval()

    # Evaluation
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.float())
            _, predicted = torch.max(outputs.data, 1)
            total = total + labels.size(0)
            correct = correct + torch.sum(predicted.eq(labels)).item()

    print(f'Accuracy: {100 * correct / total}%')


# Call the function to train and evaluate the model
# train_and_evaluate_model()
