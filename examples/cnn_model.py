import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.fc1(x))
        return x

def train_CNN_model(trial, number_parameters=16, freq_range='Beta', epochs=10, batch_size=32):
    
    base_dir = os.getcwd()
    data = pd.read_csv(base_dir + f"/dataset/{trial}.csv") 
    
    # Assuming 'X' contains your features (The frequency ranges) and 'y' contains corresponding labels
    if number_parameters == 16:
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']].values
        y = data['Image'].values
    elif number_parameters == 8:
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        y = data['Image'].values
    elif number_parameters == 4 and freq_range == 'Beta':
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].values
        y = data['Image'].values
    elif number_parameters == 4 and freq_range == 'Alpha':
        X = data[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        y = data['Image'].values
    else:
        print("Invalid number of parameters or frequency range specified.")
        return None
    
    # Assuming X is your feature data and y are your labels
    # Reshape the data to fit the model
    X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

    # Convert to torch tensors
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    # Create a CNN model
    model = CNNModel()

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()

    return model

def predict_with_cnn_model(cnn_model, features_for_model):
    # Assuming features_for_model is a 2D numpy array
    features_for_model = features_for_model.reshape(features_for_model.shape[0], 1, features_for_model.shape[1], features_for_model.shape[2])
    features_for_model = torch.from_numpy(features_for_model).float()
    outputs = cnn_model(features_for_model)
    _, predicted = torch.max(outputs, 1)
    return predicted