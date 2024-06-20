import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

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
        
        #The evluation in different ways
        accuracy = accuracy_score(y_val, y_pred)
        confusion = confusion_matrix(y_val, y_pred)
        precision = precision_score(y_val, y_pred, average='weighted')
        recall = recall_score(y_val, y_pred, average='weighted')
        f1 = f1_score(y_val, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_val, y_pred, multi_class='ovr')
        
        return accuracy, confusion, precision, recall, f1, roc_auc
   
    
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

    paras = [16, 8, 4, 4]
    
    for trial in trials:
        rf_model = RandomForestModel(n_estimators=100)

        # Get the last part of the path (file name) and remove the ".csv" extension
        trial_name = os.path.splitext(os.path.basename(trial))[0]

        for i in range(4):
            if paras[i] == 4 and i == 2:
                acc, confusion, precision, recall, f1, roc_auc = rf_model.train(trial, number_parameters=paras[i], freq_range='Alpha')
            else:
                acc, confusion, precision, recall, f1, roc_auc = rf_model.train(trial, number_parameters=paras[i])
            print(f"Accuracy for {trial_name} with {paras[i]} parameters: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}, ROC AUC: {roc_auc}")
    
