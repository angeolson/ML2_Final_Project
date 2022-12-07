# code adapted from https://towardsdatascience.com/how-to-load-a-custom-image-dataset-on-pytorch-bf10b2c529e0

# imports
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision import datasets, transforms
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

# set global vars
PATH = os.getcwd() + '/Data/Vegetable Images'

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
        data = os.listdir(train_dir + "/" + files[i])
        label = files[i]
        data_dict = {'image': data, 'label_string': ([label]*len(data)), 'label': ([label]*len(data))}
        frame = pd.DataFrame(data_dict)
        df = df.append(frame)
    encoded = pd.get_dummies(df, columns=['label'])
    encoded['target'] = encoded.iloc[:, 2:].apply(list, axis=1)
    return encoded

# create splits and encode:
train_df = getFrame(train_dir)
val_df = getFrame(val_dir)
test_df = getFrame(test_dir)

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

        label = self.data_frame.iloc[idx]['label_string']
        img_name = self.data_frame.iloc[idx]['image']
        img_path = os.path.join(self.root_dir, label, img_name)
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

# INSTANTIATE THE OBJECT
train = ImagesDataset(
    data_frame=train_df,
    root_dir=train_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)

# plot an example of an image
temp_img, temp_lab = train[0]
plt.imshow(temp_img.numpy().transpose((1, 2, 0)))
plt.title(temp_lab)
plt.axis('off')
plt.show()