import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import accuracy_score, roc_curve, classification_report, confusion_matrix, auc




def get_yhat_in_prob_array(y_pred):
    y_hat = []
    for prob in y_pred:
        y_hat.append(prob[1])
    y_hat = np.array(y_hat)
    
    return y_hat


def build_clf_report(y_true, y_hat, threshold):
    y_pred_binary = np.where(y_hat >= threshold, 1, 0)
    print(classification_report(y_true, y_pred_binary, target_names=['Not Goal', 'Goal']))



def plot_roc_auc(y_true,y_hat,threshold):
    y_pred_binary = np.where(y_hat >= threshold, 1, 0)
    fpr, tpr, _ = roc_curve(y_true, y_pred_binary)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr,color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    

def plot_confusion_matrix(y_true, y_pred, threshold):
    y_pred_binary = np.where(y_pred >= threshold, 1, 0)
    cm = confusion_matrix(y_true, y_pred_binary)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicció")
    plt.ylabel("Real")
    plt.show()
    
    
    
    
def plot_roc_aucc(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    

def plot_roc_curve1(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Corba ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Ratio falsos positius')
    plt.ylabel('Ratio veritables positius')
    #plt.title('Corba ROC')
    plt.legend(loc="lower right")

    plt.show()


