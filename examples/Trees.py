import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

def train_extra_trees_model(trial, number_parameters=16, freq_range='Beta'):
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Encode labels
    
    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing the features
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    
    # Train Extra Trees model
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_normalized, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val_normalized)
    accuracy = accuracy_score(y_val, y_pred)
    
    return model, accuracy, label_encoder



def train_gb_model(trial, number_parameters=16, freq_range='Beta'):
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
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Encode labels
    
    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Normalizing the features
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)
    
    # Train Gradient Boosting model
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model.fit(X_train_normalized, y_train)

    # Evaluate the model
    y_pred = model.predict(X_val_normalized)
    accuracy = accuracy_score(y_val, y_pred)
    
    return model, accuracy, label_encoder

def predict_extra_trees_model(model, scaler, label_encoder, features_for_model):
    features_for_model_normalized = scaler.transform(features_for_model)
    y_pred = model.predict(features_for_model_normalized)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    return y_pred_labels

def predict_gb_model(model, scaler, label_encoder, features_for_model):
    features_for_model_normalized = scaler.transform(features_for_model)
    y_pred = model.predict(features_for_model_normalized)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    return y_pred_labels


if __name__ == "__main__":
    trial = "without_individuals/pic_e_close_motion"
    paras = [16, 8, 4, 4]
    trial_name = os.path.splitext(os.path.basename(trial))[0]

    print("-" * 50, "\n\n Extra Trees \n\n", "-" * 50, "\n\n")
    
    for i in range(4):
        if paras[i] == 4 and i == 2:
            model, acc, label_encoder = train_extra_trees_model(trial, number_parameters=paras[i], freq_range='Beta')
        else:
            model, acc, label_encoder = train_extra_trees_model(trial, number_parameters=paras[i])
        print(f"Accuracy for {trial_name} with {paras[i]} parameters: {acc}")
    
    print("-" * 50, "\n\n Gradient Boosting \n\n", "-" * 50, "\n\n")    
    
    for i in range(4):
        if paras[i] == 4 and i == 2:
            model, acc, label_encoder = train_gb_model(trial, number_parameters=paras[i], freq_range='Beta')
        else:
            model, acc, label_encoder = train_gb_model(trial, number_parameters=paras[i])
        print(f"Accuracy for {trial_name} with {paras[i]} parameters: {acc}")



    
