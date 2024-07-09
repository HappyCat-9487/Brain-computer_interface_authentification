import torch
from torch import nn
import torch.nn.functional as F
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt

class CNNModel(nn.Module):
    def __init__(self, input_size=16, num_classes=6):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * (input_size//4), 32)
        self.fc2 = nn.Linear(32, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * x.size(2))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
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
    
    elif number_parameters == 8:
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values

    elif number_parameters == 4 and freq_range == 'Beta':
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].values
    
    elif number_parameters == 4 and freq_range == 'Alpha':
        X = data[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
    
    else:
        print("Invalid number of parameters or frequency range specified.")
        return None
    
    
    y = data['Image'].values
    y = pd.get_dummies(y).values  # One-hot encode labels
    
    # Splitting the data into training and validation sets (adjust test_size and random_state as needed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing the features (scaling between 0 and 1)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    
    # Reshape the data to fit the model
    X_train_normalized = X_train_normalized.reshape(X_train_normalized.shape[0], 1, X_train_normalized.shape[1])
    X_val_normalized = X_val_normalized.reshape(X_val_normalized.shape[0], 1, X_val_normalized.shape[1])
     
    
    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

    # Create Tensor datasets and Data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Our CNN model
    model = CNNModel(input_size=number_parameters, num_classes=y_train.shape[1])

    # Define the loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Train the model
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, torch.max(y_batch, 1)[1])
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_tensor)
        _, predicted = torch.max(outputs, 1)
        _, y_val_classes = torch.max(y_val_tensor, 1)
        
        classes = torch.unique(y_val_classes)
        
        #Calculate the metrics
        accuracy = (predicted == y_val_classes).sum().item() / len(y_val_classes)
        confusion = confusion_matrix(y_val_classes, predicted, labels=classes.numpy())
        precision = precision_score(y_val_classes, predicted, average='macro', zero_division=1)
        recall = recall_score(y_val_classes, predicted, average='macro')
        f1 = f1_score(y_val_classes, predicted, average='macro')
        
        # Debug: Print the shapes of the tensors
        #print("outputs shape:", outputs.shape)
        #print("y_val_tensor shape:", y_val_tensor.shape)
        
        # Compute precision-recall curve for each class
        precision_recall = {}
        for i in range(len(classes)):
            precision_p, recall_p, _ = precision_recall_curve(y_val_tensor[:, i], outputs[:, i])
            precision_recall[classes[i].item()] = (precision_p, recall_p)
        
        
    return model, accuracy, confusion, precision, recall, f1, precision_recall, classes


#todo: modify the predict function
def predict_with_cnn_model(cnn_model, features_for_model):
    features_for_model = features_for_model.reshape(features_for_model.shape[0], 1, features_for_model.shape[1])
    features_for_model = torch.from_numpy(features_for_model).float()
    
    # Ensure the model is in evaluation mode
    cnn_model.eval()  
    with torch.no_grad():
        outputs = cnn_model(features_for_model)
        _, predicted = torch.max(outputs, 1)
    return predicted



def plot_confusion_matrix(trial_name, paras, batch_size, learning_rate, epoch, confusion, classes, accuracy, precision, recall, f1):
    
    plt.figure(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[f'class {int(cls)}' for cls in classes])
    disp.plot()
    plt.suptitle(f"Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    plt.title(f"parameters: {paras} ,  batch size: {batch_size},\n lr: {learning_rate}, epoch: {epoch}", fontsize=8, pad=15, ha='center')
    
    plt.xticks(rotation=45)
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.6, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/cnn"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    
    filename = f"{trial_name}_{paras}_{batch_size}_{learning_rate}_{epoch}_confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, filename))
    #plt.show()
    
    
 
def plot_precision_recall_curve(precision_recall, trial_name, paras, batch_size, learning_rate, epoch, accuracy, precision, recall, f1):
   
    plt.figure(figsize=(13, 8))
    for label, (precision_p, recall_p) in precision_recall.items():
            plt.plot(recall_p, precision_p, lw=2, label=f'class {label}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.suptitle(f"Precision-Recall Curve: {trial_name}", y=0.98, color='black', fontsize=14, ha='center')
    plt.title(f"parameters: {paras} ,  batch size: {batch_size}, lr: {learning_rate}, epoch: {epoch}", fontsize=8, pad=15, ha='center')
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.25, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/cnn"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    
    filename = f"{trial_name}_{paras}_{batch_size}_{learning_rate}_{epoch}_precision_recall_curve.png"
    plt.savefig(os.path.join(save_dir, filename))
    #plt.show()   


#TODO: modify the main function to fit the requitements
if __name__ == "__main__":
    trial = "without_individuals/pic_e_close_motion"

    paras = [16, 8, 4, 4]

    best_params = {}
    
    # Get the last part of the path (the file name) and remove the ".csv" extension
    trial_name = os.path.splitext(os.path.basename(trial))[0]

    for i in range(4):
        if paras[i] == 4 and i == 3:
            cnn_model, acc, confusion, precision, recall, f1, precision_recall, classes = train_CNN_model(trial, number_parameters=paras[i], freq_range='Alpha')
        else:
            cnn_model, acc, confusion, precision, recall, f1, precision_recall, classes = train_CNN_model(trial, number_parameters=paras[i])
        print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
        
        # Composite score
        score = 0.6 * f1 + 0.4 * acc
        
        if score > best_score:
            best_score = score
            best_params[trial_name] = {
                "params": paras[i],
                "batch_size": batch_size,
                "hidden_layers": hidden_layer,
                "learning_rate": learning_rate,
                "epochs": epoch,
                "model": fc_model,
                "confusion": confusion,
                "classes": classes,
                "precision_recall": precision_recall,
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "score": score
            }
                                
    best_trial = best_params[trial_name]
    print(f"Best params for {trial_name}: {best_trial}")
    
    # Plot confusion matrix
    plot_confusion_matrix(
        trial_name, best_trial["params"], best_trial["batch_size"], best_trial["hidden_layers"], best_trial["learning_rate"], best_trial["epochs"],
        best_trial["confusion"], best_trial["classes"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"], best_trial["f1"]
    )

    # Plot precision-recall curves
    plot_precision_recall_curve(
        best_trial["precision_recall"], trial_name, best_trial["params"], best_trial["batch_size"], best_trial["hidden_layers"], best_trial["learning_rate"], best_trial["epochs"],
        best_trial["accuracy"], best_trial["precision"], best_trial["recall"], best_trial["f1"]
    )               