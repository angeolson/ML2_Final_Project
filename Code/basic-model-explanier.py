# purpose: creates very basic CNN model


# imports
import torch.nn as nn
from torchvision import datasets, transforms
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch

# %% --------------------------------------- Set-Up --------------------------------------------------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
PATH = os.getcwd() + '/Data/Vegetable Images'
CHANNEL = 3
SIZE = 224 # height and width

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-2
N_EPOCHS = 5
BATCH_SIZE = 12
DROPOUT = 0.5

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
train_dir = PATH + '/train'
val_dir = PATH + '/validation'
test_dir = PATH + '/test'

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
train_df = getFrame(train_dir).reset_index(drop=True)
val_df = getFrame(val_dir).reset_index(drop=True)
test_df = getFrame(test_dir).reset_index(drop=True)

# images with error: remove
#1 PIL.UnidentifiedImageError: cannot identify image file '/home/ubuntu/Final-Project-Group4/Code/Data/Vegetable Images/train/Carrot/0214.jpg'
find = train_df[ (train_df['image'] == '0214.jpg') & (train_df['label_string'] == 'Carrot')].index
train_df.drop(index=find, inplace=True)
train_df.reset_index(drop=True)

#2 PIL.UnidentifiedImageError: cannot identify image file '/home/ubuntu/Final-Project-Group4/Code/Data/Vegetable Images/train/Carrot/0475.jpg'
find = train_df[ (train_df['image'] == '0475.jpg') & (train_df['label_string'] == 'Carrot')].index
train_df.drop(index=find, inplace=True)
train_df.reset_index(drop=True)

#3 PIL.UnidentifiedImageError: cannot identify image file '/home/ubuntu/Final-Project-Group4/Code/Data/Vegetable Images/train/Bitter_Gourd/0723.jpg'
find = train_df[ (train_df['image'] == '0723.jpg') & (train_df['label_string'] == 'Bitter_Gourd')].index
train_df.drop(index=find, inplace=True)
train_df.reset_index(drop=True)

#4 PIL.UnidentifiedImageError: cannot identify image file '/home/ubuntu/Final-Project-Group4/Code/Data/Vegetable Images/train/Capsicum/0802.jpg'
find = train_df[ (train_df['image'] == '0802.jpg') & (train_df['label_string'] == 'Capsicum')].index
train_df.drop(index=find, inplace=True)
train_df.reset_index(drop=True)

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
        getBytes = transforms.ToTensor()
        imgTensor = getBytes(image)
        R_mean, G_mean, B_mean = (torch.mean(imgTensor, dim = [1,2])).numpy()
        R_std, G_std, B_std = (torch.std(imgTensor, dim = [1,2])).numpy()

        if self.transform:
            image = self.transform(image)
            norm = transforms.Normalize([R_mean, G_mean, B_mean], [R_std, G_std, B_std])
            image = norm(image)
        return (image, label)


train = ImagesDataset(
    data_frame=train_df,
    root_dir=train_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)

validation = ImagesDataset(
    data_frame=val_df,
    root_dir=val_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)

test = ImagesDataset(
    data_frame=test_df,
    root_dir=test_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)

# set to data loaders
train_loader = DataLoader(train, batch_size=BATCH_SIZE)
valid_loader = DataLoader(validation, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

# %% -------------------------------------- CNN Class ------------------------------------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # set parameters:
        self.init_inchannels = CHANNEL
        self.init_hw = SIZE
        self.init_outchannels = 16
        self.kernel = 3
        self.pool_kernel = 2
        self.stride = 1
        self.pad = 1
        self.sec_outchannels = 32
        self.n_classes = 15
        self.linear_output = 400
        self.dilation = 1

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

        # Linear 1
        self.linear1 = nn.Linear(int(self.sec_outchannels * (self.o2p2) * (self.o2p2)),self.n_classes)  # input will be flattened to (n_examples, 32 * 5 * 5)
        self.linear1_bn = nn.BatchNorm1d(self.n_classes)
        self.act = torch.relu

    def forward(self, x):
        x = self.pool1(self.convnorm1(self.act(self.conv1(x))))
        x = self.pool2(self.convnorm2(self.act(self.conv2(x))))
        x = self.linear1_bn(self.act(self.linear1(x.view(len(x), -1))))
        return x

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = CNN().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(N_EPOCHS):
    train_losses = []
    train_acc = 0.0
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels.float())
        loss.backward()
        train_losses.append(loss.item())
        accuracy = multi_accuracy_score(output, labels)
        train_acc += accuracy
        optimizer.step()

    val_losses = []
    val_acc = 0.0
    model.eval()
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        val_loss = criterion(output, labels.float())
        val_losses.append(val_loss.item())
        accuracy = multi_accuracy_score(output, labels)
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = train_acc / len(train_loader.dataset)
    epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc * 100} val_accuracy : {epoch_val_acc * 100}')

print('Done!')




import shap
explainer = shap.KernelExplainer(model.predict,X_train)