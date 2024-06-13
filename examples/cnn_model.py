import torch
from torch import nn

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

def train_CNN_model(X, y, epochs=10, batch_size=32):
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