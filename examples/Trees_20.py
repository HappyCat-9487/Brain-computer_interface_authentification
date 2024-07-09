import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, ConfusionMatrixDisplay
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import ast

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

    def train_extra_trees(self):
        model = ExtraTreesClassifier(random_state=42)
        param_grid = {
            'n_estimators': [75, 100, 200, 300],
            'max_depth': [None],
            'min_samples_split': [2, 4, 6],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        all_results = []
        for params, mean_test_score, std_test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
            best_model = ExtraTreesClassifier(random_state=42, **params)
            best_model.fit(self.X_train, self.y_train)
            y_pred = best_model.predict(self.X_val)
            y_score = best_model.predict_proba(self.X_val)
            
            classes = np.sort(np.unique(self.y_val))
            
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            
            all_results.append((params, accuracy, f1))
        
        
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
        
        return best_model, accuracy, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, all_results, grid_search

    def train_gb(self):
        model = GradientBoostingClassifier(random_state=42)
        param_grid = {
            'n_estimators': [75, 100, 200, 300],
            'learning_rate': [0.001, 0.01, 0.1],
            'max_depth': [3, 5, 7, 10]
        }
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        
        all_results = []
        for params, mean_test_score, std_test_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
            best_model = GradientBoostingClassifier(random_state=42, **params)
            best_model.fit(self.X_train, self.y_train)
            y_pred = best_model.predict(self.X_val)
            y_score = best_model.predict_proba(self.X_val)
            
            classes = np.sort(np.unique(self.y_val))
            
            accuracy = accuracy_score(self.y_val, y_pred)
            f1 = f1_score(self.y_val, y_pred, average='weighted')
            
            all_results.append((params, accuracy, f1))
        
        
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
        
        return best_model, accuracy, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, all_results, grid_search

    def predict(self, model, features_for_model):
        features_for_model_normalized = self.scaler.transform(features_for_model)
        y_pred = model.predict(features_for_model_normalized)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        return y_pred_labels



# Plot confusion matrix
def plot_confusion_matrix(trial_name, treeName, paras, n_estimator, confusion, accuracy, precision, recall, f1, labels, nor=False):
    
    plt.figure(figsize=(12, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    disp.plot()
    if nor:
        plt.suptitle(f"Normalized Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
    else:
        plt.suptitle(f"Confusion Matrix: {trial_name}", y=0.98, color='black', fontsize=12, ha='center')
        
    if treeName == "Extra_Trees":
        plt.title(f"Model: {treeName}, parameters: {paras['number_parameters']}, max_depth: {paras['max_depth']}\n, min_samples_split: {paras['min_samples_split']}, min_samples_leaf: {paras['min_samples_leaf']},\nnumber of estimators: {n_estimator}", fontsize=7, pad=15, ha='center') 
    else:
        plt.title(f"Model: {treeName}, parameters: {paras['number_parameters']}, lr: {paras['lr']}\n, max_depth: {paras['max_depth']}, number of estimators: {n_estimator}", fontsize=7, pad=15, ha='center') 
    
    plt.xticks(rotation=45)
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.95, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.9, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/Trees"
    os.makedirs(save_dir, exist_ok=True)
    plt.tight_layout()
    
    if nor:
        filename = f"{trial_name}_{treeName}_{paras}_{n_estimator}_normalized_confusion_matrix.png"
    else:
        filename = f"{trial_name}_{treeName}_{paras}_{n_estimator}_confusion_matrix.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# Plot precision-recall curves
def plot_precision_recall_curve(precision_recall, trial_name, paras, treeName, n_estimator, classes, accuracy, precision, recall, f1, labels):
    plt.figure(figsize=(13, 8))
    for label in classes:
        plt.plot(precision_recall[label][1], precision_recall[label][0], lw=2, 
                 label=f'class {labels[label]}')
        
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="best")
    plt.suptitle(f"Precision-Recall Curve for {trial_name}", y=0.95, color='black', fontsize=12, ha='center')
    plt.title(f"Model: {treeName}, parameters: {paras},\nn_estimator: {n_estimator}", fontsize=10, pad=20, ha='center')
    
    # Adjust the subplot to make space for the stats text
    plt.subplots_adjust(left=0.3, right=0.95, top=0.85, bottom=0.1)
    
    # Add accuracy, precision, recall, f1 to the plot
    stats_text = f"Accuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    plt.text(-0.2, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/Trees"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_{treeName}_{paras}_{n_estimator}_precision_recall_curve.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()


# Plot trend of n_estimators
def n_estimator_trend(trial_name, treeName, plot_data):
    plt.figure(figsize=(10, 8))

    for params, acc, f1 in plot_data:
        n_estimators = params['n_estimators']
        learning_rate = params.get('learning_rate', None)
        max_depth = params['max_depth']
        
        label_acc = f'Acc: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}'
        label_f1 = f'F1: n_estimators={n_estimators}, lr={learning_rate}, max_depth={max_depth}'
        
        plt.plot(n_estimators, acc, 'o-', label=label_acc)
        plt.plot(n_estimators, f1, 'o-',label=label_f1)

    plt.xlabel('n_estimators')
    plt.ylabel('Accuracy and F1 Score')
    plt.title(f"Trend of n_estimators from {trial_name} in {treeName} model", fontsize=12, pad=15, ha='center')
    #plt.legend(loc='best')
    
    # Save the plot to the "plots" folder
    save_dir = "/Users/luchengliang/Brain-computer_interface_authentification/plots/Trees"
    os.makedirs(save_dir, exist_ok=True)
    
    filename = f"{trial_name}_{treeName}_trend_of_n_estimators.png"
    plt.savefig(os.path.join(save_dir, filename))
    plt.close()



if __name__ == "__main__":
    trials = [
        "without_individuals/imagination",
    ]
    
    paras = [16, 8, 4, 4]
    best_params_et = {}
    best_params_gb = {}
    
    
    for trial in tqdm(trials):
        trial_name = os.path.splitext(os.path.basename(trial))[0]
        best_score_et = -1  # Initialize with a low value
        best_score_gb = -1 
        best_params_et[trial_name] = None
        best_params_gb[trial_name] = None
        
        #Collect data for plotting n_estimators trend
        plot_data_et = []  
        plot_data_gb = []
        
        
        # Extra Trees 
        
        for i in range(len(paras)):
            if paras[i] == 4 and i == 2:
                trainer = TreesModel(trial, number_parameters=paras[i], freq_range='Beta')
            else:
                trainer = TreesModel(trial, number_parameters=paras[i])
            model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, all_results_et, grid_search = trainer.train_extra_trees()
            #print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            plot_data_et.extend(all_results_et)
            print(plot_data_et)

            # Composite score
            score = 0.6 * acc + 0.4 * f1
            
            if score > best_score_et:
                best_score_et = score
                best_params_et[trial_name] = {
                    "params": {
                        "number_parameters": paras[i],
                        "max_depth": grid_search.best_params_['max_depth'],
                        "min_samples_split": grid_search.best_params_['min_samples_split'],
                        "min_samples_leaf": grid_search.best_params_['min_samples_leaf']
                    },
                    "model": model,
                    "confusion": confusion,
                    "confusion_nor": confusion_nor,
                    "n_estimator": trainer.n_estimators,
                    "classes": classes,
                    "precision_recall": precision_recall,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "labels": trainer.label_encoder.classes_,
                    "score": score
                }
            
        best_trial = best_params_et[trial_name]
        
        
        # Plot confusion matrix
        plot_confusion_matrix(trial_name, "Extra_Trees", best_trial["params"], best_trial["n_estimator"], 
                                best_trial["confusion"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"]
                                , best_trial["f1"], best_trial["labels"], nor=False)
        
        plot_confusion_matrix(trial_name, "Extra_Trees", best_trial["params"], best_trial["n_estimator"], 
                                best_trial["confusion_nor"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"]
                                , best_trial["f1"], best_trial["labels"], nor=True)
        
        
        # Plot precision-recall curves
        plot_precision_recall_curve(best_trial["precision_recall"], trial_name, best_trial["params"], "Extra_Trees", 
                                    best_trial["n_estimator"], best_trial["classes"], best_trial["accuracy"], best_trial["precision"], 
                                    best_trial["recall"], best_trial["f1"], best_trial["labels"])

        # Plot trend of n_estimators for Extra Trees
        n_estimator_trend(trial_name, "Extra_Trees", plot_data_et)
       
        
        # Gradient Boosting
        
        plot_data = []
        for i in range(len(paras)):
            if paras[i] == 4 and i == 2:
                trainer = TreesModel(trial, number_parameters=paras[i], freq_range='Beta')
            else:
                trainer = TreesModel(trial, number_parameters=paras[i])
            model, acc, confusion, confusion_nor, precision, recall, f1, precision_recall, classes, all_results_gb, grid_search = trainer.train_gb()
            #print(f"{trial_name} with {paras[i]} parameters => Accuracy: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")
            
            # Composite score
            score = 0.6 * f1 + 0.4 * acc
            
            plot_data_gb.extend(all_results_gb)
            
            if score > best_score_gb:
                best_score_gb = score
                best_params_gb[trial_name] = {
                    "params": {
                        "number_parameters": paras[i],
                        "lr": grid_search.best_params_['learning_rate'],
                        "max_depth": grid_search.best_params_['max_depth']
                    },
                    "model": model,
                    "confusion": confusion,
                    "confusion_nor": confusion_nor,
                    "n_estimator": trainer.n_estimators,
                    "classes": classes,
                    "precision_recall": precision_recall,
                    "accuracy": acc,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "labels": trainer.label_encoder.classes_,
                    "score": score
                }
            
        
        best_trial = best_params_gb[trial_name]
        
        
        # Plot confusion matrix
        plot_confusion_matrix(trial_name, "Gradient_Boosting", best_trial["params"], best_trial["n_estimator"], best_trial["confusion"], 
                              best_trial["accuracy"], best_trial["precision"], best_trial["recall"], best_trial["f1"], best_trial["labels"], nor=False)
        plot_confusion_matrix(trial_name, "Gradient_Boosting",  best_trial["params"], best_trial["n_estimator"], best_trial["confusion_nor"], 
                              best_trial["accuracy"], best_trial["precision"], best_trial["recall"], best_trial["f1"], best_trial["labels"], nor=True)
        
        # Plot precision-recall curves
        plot_precision_recall_curve(best_trial["precision_recall"], trial_name, best_trial["params"], "Gradient_Boosting", best_trial["n_estimator"],
                                    best_trial["classes"], best_trial["accuracy"], best_trial["precision"], best_trial["recall"], best_trial["f1"], best_trial["labels"])
        
        
        # Plot trend of n_estimators for Gradient Boosting
        n_estimator_trend(trial_name, "Gradient_Boosting", plot_data_gb)
