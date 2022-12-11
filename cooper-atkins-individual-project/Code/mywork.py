# Code within EDA.py file
# imports
from torchvision import datasets, transforms, models
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import matplotlib.pyplot as plt
import argparse
from matplotlib import image as mpimg

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
SAVE_FIGURES = False #Says we want to save graphs of the accuracy and loss of the training loop
# create splits and encode:
train_df = getFrame(train_dir).reset_index(drop=True)
val_df = getFrame(val_dir).reset_index(drop=True)
test_df = getFrame(test_dir).reset_index(drop=True)

# Get breakdown of training data distribution
print("Breakdown of Testing Data")
print(train_df.groupby('label_string')['image'].nunique())
print("Breakdown of Validation Data")
print(val_df.groupby('label_string')['image'].nunique())
print("Breakdown of Test Data")
print(test_df.groupby('label_string')['image'].nunique())

train = ImagesDataset(
    data_frame=train_df,
    root_dir=train_dir
)
# set to data loaders
train_loader = DataLoader(train, batch_size=BATCH_SIZE, drop_last=True, shuffle=True)
valid_loader = DataLoader(validation, batch_size=BATCH_SIZE, drop_last=True)

if SAVE_FIGURES is True:
    # Save Bean Image
    image = mpimg.imread(train_dir + "/Bean/0037.jpg")
    plt.title("Bean Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Bean_Image.png', bbox_inches = 'tight') #Save Bitter Gourd Image
    plt.clf()

    # Save Bitter Gourd Image
    image = mpimg.imread(train_dir + "/Bitter_Gourd/0009.jpg")
    plt.title("Bitter Gourd Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Bitter_Gourd_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Bottle Gourd Image
    image = mpimg.imread(train_dir + "/Bottle_Gourd/0009.jpg")
    plt.title("Bottle_Gourd Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Bottle_Gourd_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Brinjal Image
    image = mpimg.imread(train_dir + "/Brinjal/0009.jpg")
    plt.title("Brinjal Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Brinjal_Image.png', bbox_inches = 'tight')
    plt.clf() # 50

    # Save Broccoli Image
    image = mpimg.imread(train_dir + "/Broccoli/0009.jpg")
    plt.title("Broccoli Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Broccoli_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Cabbage Image
    image = mpimg.imread(train_dir + "/Cabbage/0009.jpg")
    plt.title("Cabbage Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Cabbage_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Capsicum Image
    image = mpimg.imread(train_dir + "/Capsicum/0009.jpg")
    plt.title("Capsicum Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Capsicum_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Carrot Image
    image = mpimg.imread(train_dir + "/Carrot/0009.jpg")
    plt.title("Carrot Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Carrot_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Cauliflower Image
    image = mpimg.imread(train_dir + "/Cauliflower/0009.jpg")
    plt.title("Cauliflower Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Cauliflower_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Cucumber Image
    image = mpimg.imread(train_dir + "/Cucumber/0009.jpg")
    plt.title("Cucumber Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Cucumber_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Papaya Image
    image = mpimg.imread(train_dir + "/Papaya/0009.jpg")
    plt.title("Papaya Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Papaya_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Potato Image
    image = mpimg.imread(train_dir + "/Potato/0009.jpg")
    plt.title("Potato Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Potato_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Pumpkin Image
    image = mpimg.imread(train_dir + "/Pumpkin/0009.jpg")
    plt.title("Pumpkin Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Pumpkin_Image.png', bbox_inches = 'tight')
    plt.clf()

    # Save Radish Image
    image = mpimg.imread(train_dir + "/Radish/0009.jpg")
    plt.title("Radish Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Radish_Image.png', bbox_inches = 'tight')
    plt.clf() #110

    # Save Tomato Image
    image = mpimg.imread(train_dir + "/Tomato/0009.jpg")
    plt.title("Tomato Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Tomato_Image.png', bbox_inches = 'tight')
    plt.clf()

# Code within benchmark_training.py
model_type = 'CNN'
SAVE_MODEL = True #Says we want to save the best model during training loop
SAVE_FIGURES = True
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 64
DROPOUT = 0.25

#Class structure was set up by Ange initially which is why the whole class does not appear here
self.thir_outchannels = 64
self.four_outchannels = 128
self.fif_outchannels = 256
self.dropout = nn.Dropout(DROPOUT)
self.softmax = nn.Softmax(dim=1)

# Convolution 2 + 3 with pool
# 2
self.conv2 = nn.Conv2d(in_channels=self.init_outchannels, out_channels=self.sec_outchannels, kernel_size=self.kernel,
                       stride=self.stride, padding=self.pad)
self.o2 = ((self.o1p1 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
self.convnorm2 = nn.BatchNorm2d(self.sec_outchannels)

# 3
self.conv3 = nn.Conv2d(in_channels=self.init_outchannels, out_channels=self.sec_outchannels,
                       kernel_size=self.kernel, stride=self.stride, padding=self.pad)
self.o3 = ((self.o2 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
self.convnorm2 = nn.BatchNorm2d(self.sec_outchannels)

# pooling
self.pool2 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
self.o2p2 = int(((self.o3 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

# Convolution 4
self.conv4 = nn.Conv2d(in_channels=self.sec_outchannels, out_channels=self.thir_outchannels, kernel_size=self.kernel,
                       stride=self.stride, padding=self.pad)
self.o4 = ((self.o2p2 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
self.convnorm3 = nn.BatchNorm2d(self.thir_outchannels)

self.pool3 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
self.o3p3 = int(((self.o4 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

# Convolution 5
self.conv5 = nn.Conv2d(in_channels=self.thir_outchannels, out_channels=self.four_outchannels, kernel_size=self.kernel,
                       stride=self.stride, padding=self.pad)
self.o5 = ((self.o3p3 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
self.convnorm4 = nn.BatchNorm2d(self.four_outchannels)

self.pool4 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
self.o4p4 = int(((self.o5 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

# Convolution 6
self.conv6 = nn.Conv2d(in_channels=self.four_outchannels, out_channels=self.fif_outchannels, kernel_size=self.kernel,
                       stride=self.stride, padding=self.pad)
self.o6 = ((self.o4p4 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
self.convnorm5 = nn.BatchNorm2d(self.fif_outchannels)

self.pool5 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
self.o5p5 = int(((self.o6 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

# Linear 1
self.linear1 = nn.Linear(int(self.fif_outchannels * (self.o5p5) * (self.o5p5)),
                         512)  # input will be flattened to (n_examples, 32 * 5 * 5)
self.linear2 = nn.Linear(in_features=512, out_features=256)
self.linear3 = nn.Linear(in_features=256, out_features=self.n_classes)
self.linear1_bn = nn.BatchNorm1d(self.n_classes)
self.act = torch.relu
# 156

def forward(self, x):
    x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
    x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
    x = self.pool3(self.convnorm3(self.act(self.conv4(x))))
    x = self.pool4(self.convnorm4(self.act(self.conv5(x))))
    x = self.pool5(self.convnorm5(self.act(self.conv6(x))))
    x = self.act(self.linear1(x.view(len(x), -1)))
    x = self.linear1_bn(self.act(self.linear3(self.act(self.linear2(x)))))
    x = self.softmax(x)
    return x


    #Sets the prioritized metric to be the validation accuracy
    met_test = epoch_val_acc

    #Saves the best model (assuming SAVE_MODEL=True at start): Code based on Exam 2 model saving code
    if met_test > met_test_best and SAVE_MODEL:
           torch.save(model.state_dict(), "model_benchmark.pt")
           print("The model has been saved!")
           met_test_best = met_test
torch.save(model.state_dict(), "model_main.pt")
#172
if SAVE_FIGURES is True:
    #Plots test vs train accuracy by epoch number
    plt.plot(range(epoch+1), epoch_tr_acc, label = "Train")
    plt.plot(range(epoch+1), epoch_vl_acc, label = "Test")
    plt.legend()
    plt.show()
    plt.savefig('accuracy_fig_benchmark.png', bbox_inches = 'tight')

    #Clears plot so loss doesn't also show accuracy
    plt.clf()

    #Plots test vs train loss by epoch number
    plt.plot(range(epoch+1), epoch_tr_loss, label = "Train")
    plt.plot(range(epoch+1), epoch_vl_loss, label = "Test")
    plt.legend()
    plt.show()
    plt.savefig('loss_fig_benchmark.png', bbox_inches = 'tight')

plt.savefig('accuracy_fig_main.png', bbox_inches = 'tight')
plt.savefig('loss_fig_main.png', bbox_inches = 'tight')
# Within post_hoc.py file
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
# 200
# Main Model Testing
# Import
df = pd.read_csv(CODE_PATH + "/main_test_predictions.csv")

# clean labels
df['real_labels'] = df['real_labels'].apply(cleanLabel)
df['pred_labels'] = df['pred_labels'].apply(cleanLabel)

#Get rounded predictions
df['rounded_preds'] = df['pred_labels'].apply(getRoundedPreds)

#Convert predictions to strings
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

# Within main_training.py

    #Sets the prioritized metric to be the validation accuracy
    met_test = epoch_val_acc

    #Saves the best model (assuming SAVE_MODEL=True at start): Code based on Exam 2 model saving code
    if met_test > met_test_best and SAVE_MODEL:
           torch.save(model.state_dict(), "model_main.pt")
           print("The model has been saved!")
           met_test_best = met_test

if SAVE_FIGURES is True:
    #Plots test vs train accuracy by epoch number
    plt.plot(range(epoch+1), epoch_tr_acc, label = "Train")
    plt.plot(range(epoch+1), epoch_vl_acc, label = "Test")
    plt.legend()
    plt.show()
    plt.savefig('accuracy_fig_main.png', bbox_inches = 'tight')

    #Clears plot so loss doesn't also show accuracy
    plt.clf()

    #Plots test vs train loss by epoch number
    plt.plot(range(epoch+1), epoch_tr_loss, label = "Train")
    plt.plot(range(epoch+1), epoch_vl_loss, label = "Test")
    plt.legend()
    plt.show()
    plt.savefig('loss_fig_main.png', bbox_inches = 'tight')
# 254
