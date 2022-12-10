# imports
import torch.nn as nn
from torchvision import datasets, transforms, models
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import argparse
import cv2
from matplotlib import image as mpimg
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
DATA_PATH = PATH + os.path.sep + 'Code/Data/Vegetable Images'
CODE_PATH = PATH + os.path.sep + 'Code'

classes = ['Bean', 'Bitter_Gourd', 'Bottle_Gourd', 'Brinjal', 'Broccoli', 'Cabbage', 'Capsicum', 'Carrot', 'Cauliflower', 'Cucumber', 'Papaya', 'Potato', 'Pumpkin', 'Radish', 'Tomato']

# Helper Functions:
def cleanLabel(row):
    '''
    cleans function after loading from csv file
    :param row:
    :return:
    '''
    one = row.replace('[', '').replace(']', '')
    two =  list(one.split(','))
    three = [float(item) for item in two]
    return three

# create column of rounded predictions
def getRoundedPreds(row):
    new_preds = np.zeros(15)
    new_preds[np.argmax(row)] = 1
    return list(new_preds)

# create string labels for test and predicted
def getStringLabel(row):
    idx = np.argmax(row)
    label = classes[idx]
    return label

# Main Model Testing
# Import
df = pd.read_csv(CODE_PATH + "/main_test_predictions.csv")

# clean labels
df['real_labels'] = df['real_labels'].apply(cleanLabel)
df['pred_labels'] = df['pred_labels'].apply(cleanLabel)

#Get rounded predictions
df['rounded_preds'] = df['pred_labels'].apply(getRoundedPreds)

#Convert predictions to strings
#df['true_string'] = df['target'].apply(getStringLabel)
df['true_string'] = df['label_string']
df['pred_string'] = df['pred_labels'].apply(getStringLabel)

# classification report
report = classification_report(df['true_string'], df['pred_string'])
print(report)

# Building heatmap confusion matrix
cm = confusion_matrix(df['true_string'], df['pred_string'])
cm_df = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (9,9))
sns.heatmap(cm_df, annot = True)
plt.title("Main Model Confusion Matrix")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()
plt.savefig('Main_Confusion_Matrix.png', bbox_inches = 'tight')
plt.clf()


# Benchmark Model Testing
# Import
df = pd.read_csv(CODE_PATH + "/benchmark_test_predictions.csv")

# clean labels
df['real_labels'] = df['real_labels'].apply(cleanLabel)
df['pred_labels'] = df['pred_labels'].apply(cleanLabel)

#Get rounded predictions
df['rounded_preds'] = df['pred_labels'].apply(getRoundedPreds)

#Convert predictions to strings
#df['true_string'] = df['target'].apply(getStringLabel)
df['true_string'] = df['label_string']
df['pred_string'] = df['pred_labels'].apply(getStringLabel)

print(df.head())

# classification report
report = classification_report(df['true_string'], df['pred_string'])
print(report)

# Building heatmap confusion matrix
cm = confusion_matrix(df['true_string'], df['pred_string'])
cm_df = pd.DataFrame(cm, index = classes, columns = classes)
plt.figure(figsize = (9,9))
sns.heatmap(cm_df, annot = True)
plt.title("Main Model Confusion Matrix")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()
plt.savefig('Benchmark_Confusion_Matrix.png', bbox_inches = 'tight')
plt.clf()