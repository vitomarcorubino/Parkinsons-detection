import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import librosa  # for audio processing
import glob


# Define your dataset
class AudioDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_data():
    # Define your directories
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
        audio, sample_rate = librosa.load(file, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)

        X.append(mfccs_processed)

        # Add labels based on the audio file
        if file in young_healthy_files:
            y.append(0)
        elif file in elderly_healthy_files:
            y.append(1)
        else:
            y.append(2)

    return np.array(X), np.array(y)

X, y = load_data()

X, y = load_data()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
train_data = AudioDataset(X_train, y_train)
test_data = AudioDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16, shuffle=True)


# Define the model
class AudioClassifier(nn.Module):
    def __init__(self):
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Linear(40, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 3)  # Adjust the output size to 3

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.softmax(self.layer3(x), dim=1)
        return x


model = AudioClassifier()

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):  # 50 epochs
    for i, (inputs, labels) in enumerate(train_loader):
        outputs = model(inputs.float())
        labels = labels.long()  # Convert labels to LongTensor
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{50} Loss: {loss.item()}')

# Evaluation
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total}%')
