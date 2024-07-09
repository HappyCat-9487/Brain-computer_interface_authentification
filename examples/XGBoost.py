import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

class XGBoostModel:
    def __init__(self, trial, n_estimators=100, number_parameters=16, freq_range='Beta'):
        self.n_estimators = n_estimators
        self.trial = trial
        self.number_parameters = number_parameters
        self.freq_range = freq_range
        self.base_dir = os.getcwd()
        self.data = pd.read_csv(self.base_dir + f"/dataset/{self.trial}.csv") 
        self.X, self.y = self.prepare_data()
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        self.scaler = MinMaxScaler()
        self.X_train, self.X_val, self.y_train, self.y_val = self.split_and_scale_data()
 
        
    def prepare_data(self):
        if self.number_parameters == 16:
            X = self.data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                           'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                           'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                           'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']].values
        elif self.number_parameters == 8:
            X = self.data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                           'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        elif self.number_parameters == 4 and self.freq_range == 'Beta':
            X = self.data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].values
        elif self.number_parameters == 4 and self.freq_range == 'Alpha':
            X = self.data[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        else:
            raise ValueError("Invalid number of parameters or frequency range specified.")
        
        y = self.data['Image'].values
        return X, y


    def split_and_scale_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_val_normalized = self.scaler.transform(X_val)
        return X_train_normalized, X_val_normalized, y_train, y_val


    def train_xgboost(self):
        model = XGBClassifier(n_estimators=self.n_estimators, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        y_score = model.predict_proba(self.X_val)
        
        classes = np.sort(np.unique(self.y_val))
        
        accuracy = accuracy_score(self.y_val, y_pred)
        confusion = confusion_matrix(self.y_val, y_pred, labels=classes)
        confusion_nor = confusion_matrix(self.y_val, y_pred, labels=classes, normalize='true')
        precision = precision_score(self.y_val, y_pred, average='weighted')
        recall = recall_score(self.y_val, y_pred, average='weighted')
        f1 = f1_score(self.y_val, y_pred, average='weighted')
        precision_recall = {}
        for i in range(len(classes)):
            precision_p, recall_p, _ = precision_recall_curve(self.y_val == classes[i], y_score[:, i])
            precision_recall[classes[i]] = (precision_p, recall_p)
        
        return model, accuracy, confusion, confusion_nor, precision, recall, f1, precision_recall, classes


    def predict(self, model, features_for_model):
        features_for_model_normalized = self.scaler.transform(features_for_model)
        y_pred = model.predict(features_for_model_normalized)
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
    ]
    
    paras = [16]
    criterions = ['gini', 'entropy']
    best_params = {}
    num_estimators = 200
    
    for trial in tqdm(trials):
        
        trial_name = os.path.splitext(os.path.basename(trial))[0]

        #print("-" * 50, "\n\n XGBoost \n\n", "-" * 50, "\n\n")
        
        for i in range(len(paras)):
            if paras[i] == 4 and i == 2:
                trainer = XGBoostModel(trial, number_parameters=paras[i], freq_range='Beta')
            else:
                trainer = XGBoostModel(trial, number_parameters=paras[i])
            model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes = trainer.train_xgboost()
            print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=trainer.label_encoder.classes_)
            disp.plot()
            plt.title(f"Confusion Matrix for {trial_name} with {paras[i]} parameters")
            plt.xticks(rotation=45)
            plt.show()
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_nor, display_labels=trainer.label_encoder.classes_)
            disp.plot()
            plt.title(f"Confusion Matrix for {trial_name} with {paras[i]} parameters")
            plt.xticks(rotation=45)
            plt.show()
            
            # Plot precision-recall curves
            for label in classes:
                plt.plot(precision_recall[label][1], precision_recall[label][0], lw=2, label=f'class {trainer.label_encoder.inverse_transform([label])[0]}')
            
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="best")
            plt.title(f"Precision-Recall Curve for {trial_name} with {paras[i]} parameters")
            plt.show()

            print("\n")
