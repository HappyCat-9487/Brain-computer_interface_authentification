#%%
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import seaborn as sns
from ml import train_svmm_model
from cnn_model import train_CNN_model
from randomForest import RandomForestModel
from Trees import TreesModel
from fully_connected_model import train_FC_model
import matplotlib.pyplot as plt
import numpy as np

#%%
#Check the distribution of the data in different situations, separating the data by "Image"
file_paths = glob.glob('../dataset/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    plt.figure(figsize=(10, 6))
    plt.hist(df['Image'], bins=30, alpha=0.5)
    plt.title('Distribution of "Image" in ' + file_path)
    plt.xlabel('Image')
    plt.ylabel('The numbers of Image')
    plt.show()

#%%

#Check the average values for different channels of the data by separating the data by "Image"
file_paths = glob.glob('../dataset/without_individuals/*.csv', recursive=True)

for file_path in file_paths:
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].astype('category')
    image_data = df.groupby("Image", observed=True).mean().round(2)
    print(file_path)

    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    image_data = image_data.transpose()
    for column in image_data.columns:
        plt.scatter(image_data.index, image_data[column], label=column)

    plt.title('Average values for ' + csv_name)
    plt.xticks(rotation=45)
    plt.xlabel('Different Waves of categories and sensors')
    plt.ylabel('Average value')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()
    
#%%
#Check the average and the standard deviation values for different channels of the data by separating the data by "Image"
#Recommend not run this code
for file_path in file_paths:
    df = pd.read_csv(file_path)
    df['Image'] = df['Image'].astype('category')
    
    image_mean = df.groupby("Image", observed=True).mean().round(2)
    image_std = df.groupby("Image", observed=True).std().round(2)
    
    csv_name = os.path.splitext(os.path.basename(file_path))[0]
    
    for column in image_mean.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(image_mean.index, image_mean[column], label='Mean')
        plt.scatter(image_std.index, image_std[column], label='Std')
        plt.title('Mean and Std values for ' + column + ' in ' + csv_name)
        plt.xticks(rotation=45)
        plt.xlabel('Image')
        plt.ylabel('Value')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.violinplot(x=df['Image'], y=df[column])
        plt.title('Violin plot for ' + column + ' in ' + csv_name)
        plt.xticks(rotation=45)
        plt.show()




#%%

print("Current working directory:", os.getcwd())

#%%
#Change the path
os.chdir('/Users/luchengliang/Brain-computer_interface_authentification')
#print("Current working directory:", os.getcwd())

#Check various models performance with the fundamental hyperparameters
#The models include SVM, CNN, Fully Connected Neural Network, Random Forest, Extra Trees, Gradient Boosting Trees
trials = [
    "without_individuals/pic_e_close_motion",
    "without_individuals/pic_e_close_noun",
    "without_individuals/pic_e_open_motion",
    "without_individuals/pic_e_open_noun",
    "without_individuals/imagination",
    ]

paras = [16, 8, 4, 4]

randforest_model = RandomForestModel(n_estimators=100)

for trial in trials:
    # Get the last part of the path (the file name) and remove the ".csv" extension
    trial_name = os.path.splitext(os.path.basename(trial))[0]
    
    for i in range(4):
        if paras[i] == 4 and i == 2:
            svmm_model, scaler, acc_svm, confusion_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm = train_svmm_model(trial, number_parameters=paras[i], freq_range='Beta')
            cnn_model, acc_cnn, confusion_cnn, precision_cnn, recall_cnn, f1_cnn, roc_auc_cnn = train_CNN_model(trial, number_parameters=paras[i], freq_range='Beta')
            fc_model, acc_fc, confusion_fc, precision_fc, recall_fc, f1_fc, roc_auc_fc = train_FC_model(trial, number_parameters=paras[i], freq_range='Beta')
            rf_model, acc_rf, confusion_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf = randforest_model.train(trial, number_parameters=paras[i], freq_range='Beta')
            
            trees_model = TreesModel(trial, n_estimators=100, number_parameters=paras[i], freq_range='Beta')
            model_extree, acc_et, confusion_et, precision_et, recall_et, f1_et, roc_auc_et = trees_model.train_extra_trees()
            model_gb, acc_gb, confusion_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb = trees_model.train_gb()
            
        elif paras[i] == 4 and i == 3:
            svmm_model, scaler, acc_svm, confusion_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm = train_svmm_model(trial, number_parameters=paras[i], freq_range='Alpha')
            cnn_model, acc_cnn, confusion_cnn, precision_cnn, recall_cnn, f1_cnn, roc_auc_cnn = train_CNN_model(trial, number_parameters=paras[i], freq_range='Alpha')
            fc_model, acc_fc, confusion_fc, precision_fc, recall_fc, f1_fc, roc_auc_fc = train_FC_model(trial, number_parameters=paras[i], freq_range='Alpha')
            rf_model, acc_rf, confusion_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf = randforest_model.train(trial, number_parameters=paras[i], freq_range='Alpha')
            
            trees_model = TreesModel(trial, n_estimators=100, number_parameters=paras[i], freq_range='Alpha')
            model_extree,acc_et, confusion_et, precision_et, recall_et, f1_et, roc_auc_et = trees_model.train_extra_trees()
            model_gb, acc_gb, confusion_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb = trees_model.train_gb()
            
        else:
            svmm_model, scaler, acc_svm, confusion_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm = train_svmm_model(trial, number_parameters=paras[i])
            cnn_model, acc_cnn, confusion_cnn, precision_cnn, recall_cnn, f1_cnn, roc_auc_cnn = train_CNN_model(trial, number_parameters=paras[i])
            fc_model, acc_fc, confusion_fc, precision_fc, recall_fc, f1_fc, roc_auc_fc = train_FC_model(trial, number_parameters=paras[i])
            rf_model, acc_rf, confusion_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf = randforest_model.train(trial, number_parameters=paras[i])
            
            trees_model = TreesModel(trial, n_estimators=100, number_parameters=paras[i])
            model_extree,acc_et, confusion_et, precision_et, recall_et, f1_et, roc_auc_et = trees_model.train_extra_trees()
            model_gb, acc_gb, confusion_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb = trees_model.train_gb()
        
        
        print(f"Trial and parameters: {trial_name} with {paras[i]} parameters.")
        print("Accuracy of SVM:", acc_svm, "\n")
        print("Accuracy of CNN:", acc_cnn, "\n")
        print("Accuracy of Fully Connected Neuro Network:", acc_fc, "\n")
        print("Accuracy of Random Forest:", acc_rf, "\n")
        print("Accuracy of Exta Trees:", acc_et, "\n")
        print("Accuracy of Gradient Boosting Trees:", acc_gb, "\n")
        print("-" * 30, "\n\n")  # print a separator line
        
        metrics = {
            'SVM': [acc_svm, precision_svm, recall_svm, f1_svm, roc_auc_svm],
            'CNN': [acc_cnn, precision_cnn, recall_cnn, f1_cnn, roc_auc_cnn],
            'FC': [acc_fc, precision_fc, recall_fc, f1_fc, roc_auc_fc],
            'RF': [acc_rf, precision_rf, recall_rf, f1_rf, roc_auc_rf],
            'ET': [acc_et, precision_et, recall_et, f1_et, roc_auc_et],
            'GB': [acc_gb, precision_gb, recall_gb, f1_gb, roc_auc_gb]
        }
        
        labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        x = np.arange(len(labels))  # the label locations
        width = 0.1  # the width of the bars
        
        fig, ax = plt.subplots()
        for i, (model, values) in enumerate(metrics.items()):
            ax.bar(x - width/2 + i*width, values, width, label=model)

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Scores')
        ax.set_title('Scores by model and metric')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        fig.tight_layout()

        plt.show()
        
    
    print("-" * 50, "\n\n")  # print a separator line
    