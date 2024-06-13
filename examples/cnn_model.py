import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

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
    
    # Splitting the data into training and validation sets (adjust test_size and random_state as needed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing the features (scaling between 0 and 1)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    
    # Handle categorical labels (if applicable)
    if pd.api.types.is_string_dtype(y_train):
        y_train = pd.get_dummies(y_train).values  # One-hot encode labels
    else:
        raise ValueError("Wwe are not using one-hot encoding.")
    
    # Reshape the data to fit the model
    X = X.reshape(X_train_normalized.shape[0], 1, X_train_normalized.shape[1], X_train_normalized.shape[2])

    # Convert to torch tensors
    X = torch.from_numpy(X_train_normalized).float()
    y = torch.from_numpy(y_train).long()

    # Create Tensor datasets
    train_data = TensorDataset(X, y)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    # Create a CNN model
    model = CNNModel()

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        X_val_normalized = X_val_normalized.reshape(X_val_normalized.shape[0], 1, X_val_normalized.shape[1], X_val_normalized.shape[2])
        X_val = torch.from_numpy(X_val_normalized).float()
        outputs = model(X_val)
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_val).sum().item() / len(y_val)
    
    return model, accuracy


#todo: modify the predict function
def predict_with_cnn_model(cnn_model, features_for_model):
    # Assuming features_for_model is a 2D numpy array
    features_for_model = features_for_model.reshape(features_for_model.shape[0], 1, features_for_model.shape[1], features_for_model.shape[2])
    features_for_model = torch.from_numpy(features_for_model).float()
    outputs = cnn_model(features_for_model)
    _, predicted = torch.max(outputs, 1)
    return predicted