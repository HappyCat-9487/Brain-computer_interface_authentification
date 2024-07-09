import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib

class TreesModel:
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

        else:
            raise ValueError("Invalid number of parameters or frequency range specified.")
        
        y = self.data['Image'].values
        return X, y


    def split_and_scale_data(self):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        X_train_normalized = self.scaler.fit_transform(X_train)
        X_val_normalized = self.scaler.transform(X_val)
        return X_train_normalized, X_val_normalized, y_train, y_val


    def train_extra_trees_binary(self, target_class):
        sm = SMOTE(random_state=42)
        
        cls = self.label_encoder.transform([target_class])[0]
        #print(cls)
        
        binary_y_train = (self.y_train == cls).astype(int)
        binary_y_val = (self.y_val == cls).astype(int)
        X_train_res, y_train_res = sm.fit_resample(self.X_train, binary_y_train)
        
        model = ExtraTreesClassifier(n_estimators=self.n_estimators, max_depth=None, 
                                     min_samples_split=2, min_samples_leaf=1, random_state=42)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(self.X_val)
        y_score = model.predict_proba(self.X_val)[:, 1]
        
        accuracy = accuracy_score(binary_y_val, y_pred)
        confusion = confusion_matrix(binary_y_val, y_pred)
        confusion_nor = confusion_matrix(binary_y_val, y_pred, normalize='true')
        precision = precision_score(binary_y_val, y_pred)
        recall = recall_score(binary_y_val, y_pred)
        f1 = f1_score(binary_y_val, y_pred)
        precision_p, recall_p, _ = precision_recall_curve(binary_y_val, y_score)
        
        
        results = {
            "model": model,
            "accuracy": accuracy,
            "confusion": confusion,
            "confusion_nor": confusion_nor,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision_recall": (precision_p, recall_p)
        }
        
        return results, cls


    def predict(self, model, features_for_model):
        features_for_model_normalized = self.scaler.transform(features_for_model)
        y_pred = model.predict(features_for_model_normalized)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        return y_pred_labels
    


# Plot confusion matrix
def plot_confusion_matrix(trial_name, treeName, paras, n_estimator, confusion, accuracy, precision, recall, f1, labels, nor=False):
    
    plt.figure(figsize=(30, 15))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot()
    if nor:
        plt.suptitle(f"Normalized Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    else:
        plt.suptitle(f"Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
        
    plt.title(f"Model: {treeName} binary, {paras}, \nnumber of estimators: {n_estimator}", fontsize=7, pad=15, ha='center') 
    plt.xticks()
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.4, right=0.943, top=2.0, bottom=1.3)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.9, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/swim_in_ocean"
    os.makedirs(save_dir, exist_ok=True)
    
    if nor:
        filename = f"{trial_name}_{treeName}_binary_{labels[1]}_{paras}_{n_estimator}_normalized_confusion_matrix.png"
    else:
        filename = f"{trial_name}_{treeName}_binary_{labels[1]}_{paras}_{n_estimator}_confusion_matrix.png"
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# Plot precision-recall curves
def plot_precision_recall_curve(precision_recall, trial_name, paras, treeName, n_estimator, accuracy, precision, recall, f1, labels):
    plt.figure(figsize=(13, 8))
    
    plt.plot(precision_recall[1], precision_recall[0], lw=2, label=f'class {labels[1]}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.suptitle(f"Precision-Recall Curve for {trial_name}", y=0.95, color='black', fontsize=12, ha='center')
    plt.title(f"Model: {treeName} binary, parameters: {paras}, n_estimator: {n_estimator}", fontsize=10, pad=20, ha='center')
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.2, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/swim_in_ocean"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_{treeName}_binary_{labels[1]}_{paras}_{n_estimator}_precision_recall_curve.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

if __name__ == "__main__":
    trial = "without_individuals/imagination"
    paras = 16
    target_class = "swim_in_ocean"
    
    trial_name = os.path.splitext(os.path.basename(trial))[0]

    trainer = TreesModel(trial, number_parameters=paras)
    results, cls = trainer.train_extra_trees_binary(target_class)
    
    class_name = trainer.label_encoder.inverse_transform([cls])[0]
    print(f"{trial_name} for class '{class_name}' with {paras} parameters => Accuracy: {results['accuracy']}, Precision: {results['precision']}, Recall: {results['recall']}, F1: {results['f1']}")
        
    plot_confusion_matrix(
        trial_name, 
        "Extra_Trees", 
        {"number_parameters": paras}, 
        trainer.n_estimators, 
        results['confusion'], 
        results['accuracy'], 
        results['precision'], 
        results['recall'], 
        results['f1'], 
        labels=[f"not {class_name}", class_name]
    )
    
    plot_confusion_matrix(
        trial_name, 
        "Extra_Trees", 
        {"number_parameters": paras}, 
        trainer.n_estimators, 
        results['confusion_nor'], 
        results['accuracy'], 
        results['precision'], 
        results['recall'], 
        results['f1'], 
        labels=[f"not {class_name}", class_name],
        nor=True
    )
    
    plot_precision_recall_curve(
        results['precision_recall'],
        trial_name, 
        paras, 
        "Extra_Trees", 
        trainer.n_estimators, 
        results['accuracy'], 
        results['precision'], 
        results['recall'], 
        results['f1'], 
        labels=[f"not {class_name}", class_name]
    )

    #Save the best model
    model_path = "/Users/luchengliang/Brain-computer_interface_authentification/models/model.pkl"
    joblib.dump(results['model'], model_path)
    joblib.dump(trainer.scaler, "/Users/luchengliang/Brain-computer_interface_authentification/models/scaler.pkl")
    joblib.dump(trainer.label_encoder, "/Users/luchengliang/Brain-computer_interface_authentification/models/label_encoder.pkl")
    
    print(f"Model saved to {model_path}")
    
