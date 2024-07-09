import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class RandomForestModel:
    def __init__(self, criterion='gini', n_estimators=100):
        self.n_estimators = n_estimators
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(criterion=criterion, n_estimators=n_estimators, random_state=42)

    def train(self, trial, number_parameters=16, freq_range='Beta', test_size=0.2, random_state=42):
        base_dir = os.getcwd()
        data = pd.read_csv(base_dir + f"/dataset/{trial}.csv") 
        
        # Assuming 'X' contains your features (frequency ranges) and 'y' contains corresponding labels
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
        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
        
        # Normalize features (scale between 0 and 1)
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_val_normalized = self.scaler.transform(X_val)
        
        # Train the model
        self.model.fit(X_train_normalized, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_val_normalized)
        y_score = self.model.predict_proba(X_val_normalized)
        
        classes = np.sort(np.unique(y_val))
        
        #The evluation in different ways
        accuracy = accuracy_score(y_val, y_pred)
        confusion = confusion_matrix(y_val, y_pred, labels=classes)
        confusion_nor = confusion_matrix(y_val, y_pred, labels=classes, normalize='true')
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        precision_recall = {}
        for i in range(len(classes)):
            precision_p, recall_p, _ = precision_recall_curve(y_val == classes[i], y_score[:, i])
            precision_recall[classes[i]] = (precision_p, recall_p)
        
        return self.model, accuracy, confusion, confusion_nor, precision, recall, f1, precision_recall, classes
   
    
    def predict(self, X_new):
        # Normalize features (scale between 0 and 1)
        X_new_normalized = self.scaler.transform(X_new)
        
        # Predict
        y_pred = self.model.predict(X_new_normalized)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        return y_pred_labels
    
    
# Plot confusion matrix
def plot_confusion_matrix(trial_name, paras, criterion, n_estimator, confusion, accuracy, precision, recall, f1, labels, nor=False):
    
    plt.figure(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot()
    if nor:
        plt.suptitle(f"Normalized Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    else:
        plt.suptitle(f"Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    
    plt.title(f"parameters: {paras} ,  criterion: {criterion}, \n number of estimators: {n_estimator}", fontsize=8, pad=15, ha='center')
    plt.xticks(rotation=45)
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.9, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/randomforest"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    
    if nor:
        filename = f"{trial_name}_{paras}_{criterion}_{n_estimator}_normalized_confusion_matrix.png"
    else:
        filename = f"{trial_name}_{paras}_{criterion}_{n_estimator}_confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, filename))
    #plt.show()
    


# Plot precision-recall curves
def plot_precision_recall_curve(precision_recall, trial_name, paras, criterion, n_estimator, classes, accuracy, precision, recall, f1):
   
    plt.figure(figsize=(13, 8))
    for label in classes:
        plt.plot(precision_recall[label][1], precision_recall[label][0], lw=2, label=f'class {rf_model.label_encoder.inverse_transform([label])[0]}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.suptitle(f"Precision-Recall Curve for {trial_name}", y=0.95, color='black', fontsize=12, ha='center')
    plt.title(f"parameters: {paras}, criterion: {criterion}, n_estimator: {n_estimator}", fontsize=10, pad=20, ha='center')
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.2, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/randomforest"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_{paras}_{criterion}_{n_estimator}_precision_recall_curve.png"
    plt.savefig(os.path.join(save_dir, filename))
    #plt.show()


def n_estimator_trend(trial_name, plot_data):
    # Create a new figure
    plt.figure(figsize=(10, 8))

    # Plot the data
    for n_estimator, acc, f1, para, criterion in plot_data:
        plt.plot(n_estimator, acc, 'o-', label=f'Accuracy ({para}, {criterion})')
        plt.plot(n_estimator, f1, 'o-',label=f'F1 ({para}, {criterion})')
        

    # Add labels and a legend
    plt.xlabel('n_estimator')
    plt.ylabel('Accuracy and F1 Score')
    plt.title(f"Trend of n_estimators from {trial_name}", fontsize=12, pad=15, ha='center')
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/randomforest"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_trend_of_n_estimators.png"
    plt.savefig(os.path.join(save_dir, filename))
    #plt.show()
    

if __name__ == "__main__":
    trials = [
        "without_individuals/pic_e_close_motion",
        "without_individuals/pic_e_close_noun",
        "without_individuals/pic_e_open_motion",
        "without_individuals/pic_e_open_noun",
        "without_individuals/imagination",
    ]

    paras = [16, 8, 4, 4]
    criterions = ['gini', 'entropy']
    best_params = {}
    num_estimators = 200
    
    for trial in trials:

        # Get the last part of the path (the file name) and remove the ".csv" extension
        trial_name = os.path.splitext(os.path.basename(trial))[0]
        best_score = -1  # Initialize with a low value
        best_params[trial_name] = None
        
        plot_trend_n_estimator = []
        
        for k in tqdm(range(num_estimators)):
            for criterion in criterions:
                for i in range(len(paras)):
                    
                    rf_model = RandomForestModel(criterion=criterion, n_estimators=k+1)
                    
                    if paras[i] == 4 and i == 2:
                        model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes  = rf_model.train(trial, number_parameters=paras[i], freq_range='Alpha')
                    else:
                        model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes  = rf_model.train(trial, number_parameters=paras[i])
                    #print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
                    
                    # Add the data to the plot_data list
                    plot_trend_n_estimator.append((k+1, acc, f1, paras[i], criterion))
                    
                    # Composite score
                    score = 0.6 * f1 + 0.4 * acc
                    
                    if score > best_score:
                        best_score = score
                        best_params[trial_name] = {
                            "params": paras[i],
                            "model": model,
                            "confusion": confusion,
                            "confusion_nor": confusion_nor,
                            "criterion": criterion,
                            "n_estimator": k+1,
                            "classes": classes,
                            "precision_recall": precision_recall,
                            "accuracy": acc,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "labels": rf_model.label_encoder.classes_,
                            "score": score
                        }
                        
        best_trial = best_params[trial_name]
        #print("\n")
        #print(f"Best params for {trial_name}: {best_trial}")
                    
        # Plot confusion matrix
        plot_confusion_matrix(trial_name, best_trial["params"], best_trial["criterion"], best_trial["n_estimator"], 
                                best_trial["confusion"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"]
                                , best_trial["f1"], best_trial["labels"], nor=False)
        
        # Plot confusion matrix
        plot_confusion_matrix(trial_name, best_trial["params"], best_trial["criterion"], best_trial["n_estimator"], 
                                best_trial["confusion_nor"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"]
                                , best_trial["f1"], best_trial["labels"], nor=True)
        
        # Plot precision-recall curves
        plot_precision_recall_curve(best_trial["precision_recall"], trial_name, best_trial["params"], best_trial["criterion"], best_trial["n_estimator"], 
                                    best_trial["classes"],  best_trial["accuracy"], best_trial["precision"], best_trial["recall"], 
                                    best_trial["f1"])
        
        
        #Plot the trend of n_estimators
        n_estimator_trend(trial_name, plot_trend_n_estimator)
                    
                    
                    
                   
                

    
    
