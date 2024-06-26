import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.n_estimators = n_estimators
        self.scaler = MinMaxScaler()
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

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

if __name__ == "__main__":
    trials = [
        "without_individuals/pic_e_close_motion",
        "without_individuals/pic_e_close_noun",
        "without_individuals/pic_e_open_motion",
        "without_individuals/pic_e_open_noun",
        "without_individuals/imagination",
    ]

    paras = [16]
    
    for trial in trials:
        rf_model = RandomForestModel(n_estimators=100)

        # Get the last part of the path (file name) and remove the ".csv" extension
        trial_name = os.path.splitext(os.path.basename(trial))[0]

        for i in range(len(paras)):
            if paras[i] == 4 and i == 2:
                model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes  = rf_model.train(trial, number_parameters=paras[i], freq_range='Alpha')
            else:
                model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes  = rf_model.train(trial, number_parameters=paras[i])
            print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=rf_model.label_encoder.classes_)
            disp.plot()
            plt.title(f"Confusion Matrix for {trial_name} with {paras[i]} parameters")
            plt.xticks(rotation=45)
            plt.show()
            
            # Plot confusion matrix
            disp = ConfusionMatrixDisplay(confusion_matrix=confusion_nor, display_labels=rf_model.label_encoder.classes_)
            disp.plot()
            plt.title(f"Confusion Matrix for {trial_name} with {paras[i]} parameters")
            plt.xticks(rotation=45)
            plt.show()
            
            # Plot precision-recall curves
            for label in classes:
                plt.plot(precision_recall[label][1], precision_recall[label][0], lw=2, label=f'class {rf_model.label_encoder.inverse_transform([label])[0]}')
            
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.legend(loc="best")
            plt.title(f"Precision-Recall Curve for {trial_name} with {paras[i]} parameters")
            plt.show()
            
            print("\n")
            
        print('-'* 50)
    
    
