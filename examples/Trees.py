import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

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
        model = ExtraTreesClassifier(n_estimators=self.n_estimators, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        return model, accuracy

    def train_gb(self):
        model = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=0.1, max_depth=3, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_val)
        accuracy = accuracy_score(self.y_val, y_pred)
        return model, accuracy

    def predict(self, model, features_for_model):
        features_for_model_normalized = self.scaler.transform(features_for_model)
        y_pred = model.predict(features_for_model_normalized)
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
    
    for trial in tqdm(trials):
        paras = [16, 8, 4, 4]
        trial_name = os.path.splitext(os.path.basename(trial))[0]

        print("-" * 50, "\n\n Extra Trees \n\n", "-" * 50, "\n\n")
        
        for i in range(4):
            if paras[i] == 4 and i == 2:
                trainer = TreesModel(trial, number_parameters=paras[i], freq_range='Beta')
            else:
                trainer = TreesModel(trial, number_parameters=paras[i])
            model, acc = trainer.train_extra_trees()
            print(f"Accuracy for {trial_name} with {paras[i]} parameters: {acc}")
        
        print("-" * 50, "\n\n Gradient Boosting \n\n", "-" * 50, "\n\n")    
        
        for i in range(4):
            if paras[i] == 4 and i == 2:
                trainer = TreesModel(trial, number_parameters=paras[i], freq_range='Beta')
            else:
                trainer = TreesModel(trial, number_parameters=paras[i])
            model, acc = trainer.train_gb()
            print(f"Accuracy for {trial_name} with {paras[i]} parameters: {acc}")
