import torch
from sklearn.metrics import f1_score,classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

def metric(losses, predictions, labels, class_len):
    cmatrix = confusion_matrix(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[x for x in range(class_len)])
    accuracy = 100. * cmatrix.diagonal() / cmatrix.sum(axis=1)
    class_loss = [losses[labels.cpu()==j].mean().item() for j in range(class_len)]
    total_accuracy = 100. * cmatrix.diagonal().sum() / cmatrix.sum()
    total_loss = losses.mean().item()
    weighted_f1 = f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), average='weighted')
    class_f1 =  f1_score(labels.cpu().numpy(), predictions.cpu().numpy(), labels=[x for x in range(class_len)], average=None)
    
    return {
        "total_loss":total_loss,
        "total_accuracy": total_accuracy,
        "weighted_f1": weighted_f1,
        "class_loss": class_loss,
        "class_accuracy":accuracy,
        "class_f1":class_f1,
        "cmatrix": cmatrix}

def plot_confusion_matrix(cmatrix,save_path):
    plt.figure(figsize=(10,8))
    sns.heatmap(cmatrix, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    plt.savefig(os.path.join(save_path,"cmatrix.png"))