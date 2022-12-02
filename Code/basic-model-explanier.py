# purpose: creates very basic CNN model

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
#os.system('pip3 install shap')
import shap

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
LR = 1e-3
N_EPOCHS = 10
BATCH_SIZE = 256
n_classes = 15

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


# some images are corrupted--need to test if an image can be opened, then remove it from the dataframe before it even gets loaded.

def getRemovalList(directory):
    # code adapted from https://stackoverflow.com/questions/63754311/unidentifiedimageerror-cannot-identify-image-file
    bad_list = []
    mac_files = ['.DS_Store']
    class_list = [file for file in os.listdir(directory) if file not in mac_files]  # list of classes ie dog or cat
    for klass in class_list:  # iterate through the two classes
        class_path = os.path.join(directory, klass)  # path to class directory
        file_list = os.listdir(class_path)  # create list of files in class directory
        for f in file_list:  # iterate through the files
            fpath = os.path.join(class_path, f)
            try:
                image = Image.open(fpath)
                getBytes = transforms.ToTensor()
                imgTensor = getBytes(image)
            except:
                bad_list.append(fpath)
    return bad_list

badlist_train = getRemovalList(train_dir)
badlist_val = getRemovalList(val_dir)
badlist_test = getRemovalList(test_dir)

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
        transforms.Resize((SIZE,SIZE)),
        transforms.ToTensor()
    ])
)

test = ImagesDataset(
    data_frame=test_df,
    root_dir=test_dir,
    transform=transforms.Compose([
        transforms.Resize((SIZE,SIZE)),
        transforms.ToTensor()
    ])
)

# set to data loaders
train_loader = DataLoader(train, batch_size=BATCH_SIZE)
valid_loader = DataLoader(validation, batch_size=BATCH_SIZE)
test_loader = DataLoader(test, batch_size=BATCH_SIZE)


# %% -------------------------------------- CNN Class ------------------------------------------------------------------
def model_definition(n_classes):
    '''
        Define a Keras sequential model
        Compile the model
    '''

    model = models.resnet50(weights="IMAGENET1K_V2")
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)

    model = model.to(device)

    return model

# %% -------------------------------------- Training Prep ----------------------------------------------------------
model = model_definition(n_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

# %% -------------------------------------- Training Loop ----------------------------------------------------------
print("Starting training loop...")
valid_loss_min = np.Inf
epoch_tr_loss, epoch_vl_loss = [], []
epoch_tr_acc, epoch_vl_acc = [], []

for epoch in range(N_EPOCHS):
    train_losses = []
    train_acc = []
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels.float())
        loss.backward()
        train_losses.append(loss.item())
        #accuracy = multi_accuracy_score(output, labels)
        accuracy = accuracy_score(y_true=labels.cpu().numpy().astype(int),
                                  y_pred=output.detach().cpu().numpy().astype(int))
        train_acc += accuracy
        optimizer.step()

    val_losses = []
    val_acc = []
    model.eval()
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        val_loss = criterion(output, labels.float())
        val_losses.append(val_loss.item())
        #accuracy = multi_accuracy_score(output, labels)
        accuracy = accuracy_score(y_true=labels.cpu().numpy().astype(int),
                                  y_pred=output.detach().cpu().numpy().astype(int))
        val_acc += accuracy

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = np.mean(train_acc)
    epoch_val_acc = np.mean(val_acc)
    # epoch_train_acc = train_acc / len(train_loader.dataset)
    # epoch_val_acc = val_acc / len(valid_loader.dataset)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')

print('Done!')

# for inputs, labels in train_loader:
#     break
# explainer = shap.KernelExplainer(model, inputs)




