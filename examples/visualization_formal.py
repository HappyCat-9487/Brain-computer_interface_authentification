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


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = 'd'
    thresh = cm.max() / 2.
    
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def confusion_metrics_analysis(confusion_svm, confusion_cnn, confusion_fc, confusion_rf, confusion_et, confusion_gb):
    
    confusion_matrices = {
        'SVM': confusion_svm,
        'CNN': confusion_cnn,
        'FC': confusion_fc,
        'RF': confusion_rf,
        'ET': confusion_et,
        'GB': confusion_gb
    }

    #print(confusion_matrices)
    
    
    labels_dict = {
        'SVM': confusion_svm[1],
        'CNN': confusion_cnn[1],
        'FC': confusion_fc[1],
        'RF': confusion_rf[1],
        'ET': confusion_et[1],
        'GB': confusion_gb[1]
    }
    
    # Assuming all confusion matrices have the same class labels
    #classes = labels_dict[model_name]
    #print(classes)
    
    for model_name, confusion in confusion_matrices.items():
        #plt.figure(figsize=(10, 7))
        print(model_name)
        print(confusion)
        print(labels_dict[model_name])
        #plot_confusion_matrix(confusion, classes, title=f'Confusion matrix for {model_name}')
        #plt.show()
    

#Check various models performance with the hyperparameters
#The models include SVM, CNN, Fully Connected Neural Network, Random Forest, Extra Trees, Gradient Boosting Trees
#Get the performance of the models
def get_analysis(trial, paras):
    # Get the last part of the path (the file name) and remove the ".csv" extension
    trial_name = os.path.splitext(os.path.basename(trial))[0]
    
    randforest_model = RandomForestModel(n_estimators=100)

    for i in range(len(paras)):
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
        
        confusion_metrics_analysis(confusion_svm, confusion_cnn, confusion_fc, confusion_rf, confusion_et, confusion_gb)


if __name__ == "__main__":
    #Change the path
    os.chdir('/Users/luchengliang/Brain-computer_interface_authentification')
    #print("Current working directory:", os.getcwd())

    paras = [16]

    trial = "without_individuals/imagination"

    get_analysis(trial, paras)

