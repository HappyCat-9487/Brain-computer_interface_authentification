import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve
import numpy as np

class LightGBMModel:
    def __init__(self, trial, number_parameters=16, freq_range='Beta'):
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


    def train_lightgbm(self):
        param_grid = {
            'n_estimators': [10],
            'learning_rate': [0.1],
            'max_depth': [3]
        }
        model = LGBMClassifier(random_state=42, verbose=-1)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(self.X_val)
        y_score = best_model.predict_proba(self.X_val)
        
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
        
        return best_model, accuracy, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, grid_search.best_params_

    def predict(self, model, features_for_model):
        features_for_model_normalized = self.scaler.transform(features_for_model)
        y_pred = model.predict(features_for_model_normalized)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        return y_pred_labels

# Plot confusion matrix
def plot_confusion_matrix(trial_name, paras, dataparas, n_estimator, confusion, accuracy, precision, recall, f1, labels, nor=False):
    plt.figure(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot()
    if nor:
        plt.suptitle(f"Normalized Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    else:
        plt.suptitle(f"Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
        
    plt.title(f"Model: LightGBM, data parameters: {dataparas}, \nparameters: {paras}, \nn_estimators: {n_estimator}", fontsize=10, pad=15, ha='center')
    plt.xticks(rotation=45)
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
    
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.9, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/LightGBM"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    
    if nor:
        filename = f"{trial_name}_LightGBM_{dataparas}_{paras}_{n_estimator}_normalized_confusion_matrix.png"
    else:
        filename = f"{trial_name}_LightGBM_{dataparas}_{paras}_{n_estimator}_confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

# Plot precision-recall curves
def plot_precision_recall_curve(precision_recall, trial_name, paras, dataparas, n_estimator, classes, accuracy, precision, recall, f1, labels):
    plt.figure(figsize=(13, 8))
    for label in classes:
        plt.plot(precision_recall[label][1], precision_recall[label][0], lw=2, label=f'class {labels[label]}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.suptitle(f"Precision-Recall Curve for {trial_name}", y=0.95, color='black', fontsize=12, ha='center')
    plt.title(f"Model: LightGBM, data parameters: {dataparas}, \nparameters: {paras}, n_estimators: {n_estimator}", fontsize=10, pad=20, ha='center')
    
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.2, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/LightGBM"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_LightGBM_{dataparas}_{paras}_{n_estimator}_precision_recall_curve.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()

if __name__ == "__main__":
    trials = [
        "without_individuals/pic_e_close_motion",
    ]
    
    paras = [16]
    best_params = {}
    
    for trial in tqdm(trials):
        trial_name = os.path.splitext(os.path.basename(trial))[0]
        
        best_score = -1
        
        for i in range(len(paras)):
            trainer = LightGBMModel(trial, number_parameters=paras[i])
            model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, best_params_ = trainer.train_lightgbm()
            #print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            score = 0.6 * f1 + 0.4 * acc
            
            if score > best_score:
                best_score = acc
                best_params[trial_name] = {
                    "params": best_params_,
                    "dataParams": paras[i],
                    "n_estimator": model.n_estimators,
                    "confusion": confusion,
                    "confusion_nor": confusion_nor,
                    "precision_recall": precision_recall,
                    "classes": classes,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "labels": trainer.label_encoder.inverse_transform(classes)
                }

    for trial in trials:
        trial_name = os.path.splitext(os.path.basename(trial))[0]
        plot_confusion_matrix(
            trial_name, 
            best_params[trial_name]["params"],
            best_params[trial_name]["dataParams"],
            best_params[trial_name]["n_estimator"], 
            best_params[trial_name]["confusion"], 
            best_params[trial_name]["accuracy"], 
            best_params[trial_name]["precision"], 
            best_params[trial_name]["recall"], 
            best_params[trial_name]["f1"], 
            best_params[trial_name]["labels"],
            nor=False
        )
        plot_confusion_matrix(
            trial_name, 
            best_params[trial_name]["params"],
            best_params[trial_name]["dataParams"],
            best_params[trial_name]["n_estimator"], 
            best_params[trial_name]["confusion_nor"], 
            best_params[trial_name]["accuracy"], 
            best_params[trial_name]["precision"], 
            best_params[trial_name]["recall"], 
            best_params[trial_name]["f1"], 
            best_params[trial_name]["labels"],
            nor=True
        )
        plot_precision_recall_curve(
            best_params[trial_name]["precision_recall"],
            trial_name, 
            best_params[trial_name]["params"],
            best_params[trial_name]["dataParams"],
            best_params[trial_name]["n_estimator"], 
            best_params[trial_name]["classes"], 
            best_params[trial_name]["accuracy"], 
            best_params[trial_name]["precision"], 
            best_params[trial_name]["recall"], 
            best_params[trial_name]["f1"], 
            best_params[trial_name]["labels"]
        )
