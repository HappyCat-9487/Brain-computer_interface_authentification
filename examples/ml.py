'''
# Load libraries
import pandas as pd
# from pandas import read_csv
# from pandas.plotting import scatter_matrix
# from matplotlib import pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import StratifiedKFold
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.naive_bayes import GaussianNB
# from sklearn.svm import SVC
# import seaborn as sns


# Load dataset
data = pd.read_csv("./dataset/trial_James.csv")
data['Image'] = data['Image'].astype('category')
image_data = data.groupby("Image").mean().round(2)
print(image_data)
'''

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

#Just for checking if the data have some works
def check_if_data_works(base_dir):
    data = pd.read_csv(base_dir + '/dataset/Pictures/e_close/motion/trial_1.csv')
    data['Image'] = data['Image'].astype('category')
    image_data = data.groupby("Image", observed=True).mean().round(2)
    print(image_data)

    
def train_svmm_model(trial, number_parameters=16, freq_range='Beta', kernel='rbf', C=1.0, gamma='scale'):
    base_dir = os.getcwd()
    data = pd.read_csv(base_dir + f"/dataset/{trial}.csv") 
    
    # Assuming 'X' contains your features (The frequency ranges) and 'y' contains corresponding labels
    if number_parameters == 16:
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10',
                'Theta_TP9', 'Theta_AF7', 'Theta_AF8', 'Theta_TP10',
                'Delta_TP9', 'Delta_AF7', 'Delta_AF8', 'Delta_TP10']].values
        y = data['Image'].values
    elif number_parameters == 8:
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10', 
                'Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        y = data['Image'].values
    elif number_parameters == 4 and freq_range == 'Beta':
        X = data[['Beta_TP9', 'Beta_AF7', 'Beta_AF8', 'Beta_TP10']].values
        y = data['Image'].values
    elif number_parameters == 4 and freq_range == 'Alpha':
        X = data[['Alpha_TP9', 'Alpha_AF7', 'Alpha_AF8', 'Alpha_TP10']].values
        y = data['Image'].values
    else:
        print("Invalid number of parameters or frequency range specified.")
        return None

    # Splitting the data into training and validation sets (adjust test_size and random_state as needed)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalizing the features (scaling between 0 and 1)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)


    # Initializing SVM classifier (you can experiment with different kernels and parameters)
    svm_classifier = SVC(kernel=kernel, C=C, gamma=gamma)  # Example with RBF kernel

    # Training the SVM model
    svm_classifier.fit(X_train_normalized, y_train)

    # Making predictions on the validation set
    predictions = svm_classifier.predict(X_val_normalized)
    y_score = svm_classifier.predict_proba(X_val)
    
    # Calculating metrics
    accuracy = accuracy_score(y_val, predictions)
    confusion = confusion_matrix(y_val, predictions)
    precision = precision_score(y_val, predictions, average='weighted')
    recall = recall_score(y_val, predictions, average='weighted')
    f1 = f1_score(y_val, predictions, average='weighted')
    roc_auc = roc_auc_score(y_val, y_score, multi_class='ovr')

    return svm_classifier, scaler, accuracy, confusion, precision, recall, f1, roc_auc

# Remember to replace X and y with your actual feature and label data. 
# Also, feel free to adjust the SVM parameters (kernel, C, etc.) and explore different kernels (e.g., 'linear', 'poly', 'rbf') to find the best model for your data.

# Assuming 'X_new' contains new samples of Beta and Alpha frequencies for prediction

def predict_with_svmm_model(svm_classifier, scaler, X_new):
#trial testing data
    # Normalize the new data using the same scaler used for training/validation data
    X_new = np.array(X_new).reshape(1, -1)  # Reshape the data. In order to change the shape of the data.
    X_new_normalized = scaler.transform(X_new)

    # Making predictions on the new data
    new_predictions = svm_classifier.predict(X_new_normalized)
    return new_predictions
    # Displaying the predicted labels
    # print("Predicted Labels for New Data:")
    # for prediction in new_predictions:
    #     print(prediction)

if __name__ == "__main__":
    #Just for checking if the data have some works
    base_dir = os.getcwd()
    #print(base_dir)
    
    check_if_data_works(base_dir)
    
    #testing if code works:
    trial_test = "Pictures/e_close/motion/trial_1"
    svm_classifier, scaler = train_svmm_model(trial_test) #training the model 
    prediction = predict_with_svmm_model(svm_classifier, scaler, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) 
    print(prediction)
    #^^ testing the predictive model - replace 1, 1 with any betaValue, alphaValue, DeltaValue, and ThetaValue you wish

