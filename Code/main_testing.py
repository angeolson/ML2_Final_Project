
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

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
# parser = argparse.ArgumentParser()
# parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
# args = parser.parse_args()
# PATH = args.path
PATH = '/home/ubuntu/Final-Project-Group4'
DATA_PATH = PATH + os.path.sep + 'Data/Vegetable Images'
CODE_PATH = PATH + os.path.sep + 'Code'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
PATH = os.getcwd() + '/Data/Vegetable Images'
CHANNEL = 3
SIZE = 224 # height and width
n_classes = 15
model_type = 'transformer'

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
BATCH_SIZE = 64
DROPOUT = 0.25

# %% ----------------------------------- Helper Functions --------------------------------------------------------------
def multi_accuracy_score(y_pred, y_true):
    '''
    Compute accuracy for multiclass/multilabel, based on sklearn
    :param y_true:
    :param y_pred:
    :return: accuracy score
    '''
    pred = torch.round(y_pred)
    differing_labels = torch.count_nonzero(y_true - pred, axis=1)
    score = torch.sum(differing_labels == 0)

    return score

# %% -------------------------------------- Set Data ------------------------------------------------------------------
# create dataframes for train, val, and test data:

test_dir = DATA_PATH + '/test'

def getFrame(filepath):
    '''
    :param filepath:
    :return: dataframe of images and labels for the given split, encoded and converted to target
    '''
    mac_files = ['.DS_Store']
    files = [file for file in os.listdir(filepath) if file not in mac_files]
    df = pd.DataFrame()
    for i in range(len(files)): # loop through each category
        data = os.listdir(filepath + "/" + files[i])
        label = files[i]
        data_dict = {'image': data, 'label_string': ([label]*len(data)), 'label': ([label]*len(data))}
        frame = pd.DataFrame(data_dict)
        df = df.append(frame)
    encoded = pd.get_dummies(df, columns=['label'])
    encoded['target'] = encoded.iloc[:, 2:].apply(list, axis=1)
    return encoded

# create splits and encode:
test_df = getFrame(test_dir).reset_index(drop=True)

# create dataloader
class ImagesDataset(Dataset):
    """
    The Class will act as the container for our dataset. It will take your dataframe, the root path, and also the transform function for transforming the dataset.
    """

    def __init__(self, data_frame, root_dir, transform=None):
        self.data_frame = data_frame
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        # Return the length of the dataset
        return len(self.data_frame)

    def __getitem__(self, idx):
        # Return the observation based on an index. Ex. dataset[0] will return the first element from the dataset, in this case the image and the label.
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label_name = self.data_frame.iloc[idx]['label_string']
        label = self.data_frame.iloc[idx]['target']
        label = torch.Tensor(label)
        img_name = self.data_frame.iloc[idx]['image']
        img_path = os.path.join(self.root_dir, label_name, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
            norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            image = norm(image)
        return (image, label)

test = ImagesDataset(
    data_frame=test_df,
    root_dir=test_dir,
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)

# set to data loaders
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # set parameters:
        self.init_inchannels = CHANNEL
        self.init_hw = SIZE
        self.init_outchannels = 16
        self.kernel = 3 # Testing kernel size of 4 instead of 3
        self.pool_kernel = 2
        self.stride = 1
        self.pad = 1
        self.sec_outchannels = 32
        self.n_classes = 15
        self.linear_output = 400
        self.dilation = 1
        self.thir_outchannels = 64
        self.four_outchannels = 128
        self.fif_outchannels = 256
        self.dropout = nn.Dropout(DROPOUT)

        # Convolution 1
        self.conv1 = nn.Conv2d(in_channels=self.init_inchannels, out_channels=self.init_outchannels, kernel_size=self.kernel, stride=self.stride, padding=self.pad)
        self.o1 = int(((self.init_hw + 2*self.pad - (self.dilation*(self.kernel-1)) -1)/self.stride) + 1)
        self.convnorm1 = nn.BatchNorm2d(self.init_outchannels)
        self.pool1 = nn.MaxPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
        self.o1p1 = int(((self.o1 + 2*self.pad - (self.dilation*(self.pool_kernel-1)) -1)/self.pool_kernel) + 1)

        # Convolution 2 + 3 with pool
        #2
        self.conv2 = nn.Conv2d(in_channels=self.init_outchannels, out_channels=self.sec_outchannels,kernel_size=self.kernel, stride=self.stride, padding=self.pad)
        self.o2 = ((self.o1p1 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
        self.convnorm2 = nn.BatchNorm2d(self.sec_outchannels)

        #3
        self.conv3 = nn.Conv2d(in_channels=self.init_outchannels, out_channels=self.sec_outchannels,
                               kernel_size=self.kernel, stride=self.stride, padding=self.pad)
        self.o3 = ((self.o2 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
        self.convnorm2 = nn.BatchNorm2d(self.sec_outchannels)


        #pooling
        self.pool2 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
        self.o2p2 = int(((self.o3 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)


        # Convolution 4
        self.conv4 = nn.Conv2d(in_channels=self.sec_outchannels, out_channels=self.thir_outchannels, kernel_size=self.kernel, stride=self.stride,padding=self.pad)
        self.o4 = ((self.o2p2 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
        self.convnorm3 = nn.BatchNorm2d(self.thir_outchannels)

        self.pool3 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
        self.o3p3 = int(((self.o4 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

        # Convolution 5
        self.conv5 = nn.Conv2d(in_channels=self.thir_outchannels, out_channels=self.four_outchannels, kernel_size=self.kernel, stride=self.stride,padding=self.pad)
        self.o5 = ((self.o3p3 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
        self.convnorm4 = nn.BatchNorm2d(self.four_outchannels)

        self.pool4 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
        self.o4p4 = int(((self.o5 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

        # Convolution 6
        self.conv6 = nn.Conv2d(in_channels=self.four_outchannels, out_channels=self.fif_outchannels, kernel_size=self.kernel, stride=self.stride,padding=self.pad)
        self.o6 = ((self.o4p4 + 2 * self.pad - (self.dilation * (self.kernel - 1)) - 1) / self.stride) + 1
        self.convnorm5 = nn.BatchNorm2d(self.fif_outchannels)

        self.pool5 = nn.AvgPool2d(kernel_size=self.pool_kernel, stride=self.pool_kernel, padding=self.pad)
        self.o5p5 = int(((self.o6 + 2 * self.pad - (self.dilation * (self.pool_kernel - 1)) - 1) / self.pool_kernel) + 1)

        # Linear 1
        self.linear1 = nn.Linear(int(self.fif_outchannels * (self.o5p5) * (self.o5p5)),512)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear2 = nn.Linear(in_features = 512,out_features = 256)
        self.linear3 = nn.Linear(in_features = 256, out_features = self.n_classes)
        self.linear1_bn = nn.BatchNorm1d(self.n_classes)
        self.act = torch.relu


    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.pool3(self.convnorm3(self.act(self.conv4(x))))
        x = self.pool4(self.convnorm4(self.act(self.conv5(x))))
        x = self.pool5(self.convnorm5(self.act(self.conv6(x))))
        x = self.act(self.linear1(x.view(len(x), -1)))
        x = self.linear1_bn(self.act(self.linear3(self.act(self.linear2(x)))))
        return x
# %% -------------------------------------- Transformer ----------------------------------------------------------
# Note: currently only the transformer works
transformer = models.resnet34(pretrained=True)
for param in transformer.parameters():
    param.requires_grad = False
for param in transformer.layer4.parameters():
    param.requires_grad = True

transformer.fc = nn.Sequential(
    nn.Linear(transformer.fc.in_features, n_classes),
    nn.Softmax(dim=1)
                               )

# %% -------------------------------------- Testing Prep ----------------------------------------------------------
if model_type == 'CNN':
    model = CNN().to(device)
else:
    model = transformer.to(device)

model.load_state_dict(torch.load('model_main.pt', map_location=device))
model.to(device)

criterion = nn.BCELoss()

# %% -------------------------------------- Testing Loop ----------------------------------------------------------
print("Starting training loop...")
met_test = 0
met_test_best = 0

test_losses = []
test_acc = []
final_pred_labels = []
final_real_labels = []
model.eval()
for inputs, labels in test_loader:
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs)
    test_loss = criterion(output, labels.float())
    test_losses.append(test_loss.item())
    preds = output.detach().cpu().numpy()
    new_preds = np.zeros(preds.shape)
    for i in range(len(preds)):
        new_preds[i][np.argmax(preds[i])] = 1
    accuracy = accuracy_score(y_true=labels.cpu().numpy().astype(int),
                                  y_pred=new_preds.astype(int))
    test_acc.append(accuracy)
    for i in range(len(preds)):
        result_list = [e for e in preds[i]]
        label_list = [e for e in labels.cpu().numpy()[i]]
        final_pred_labels.append(result_list)
        final_real_labels.append(label_list)

test_loss_av = np.mean(test_losses)
test_acc_av = np.mean(test_acc)
print(f'Test Accuracy: {test_acc_av} Test Loss: {test_loss_av}')
test_df['pred_labels'] = final_pred_labels
test_df['real_labels'] = final_real_labels
