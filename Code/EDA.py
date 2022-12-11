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
parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
DATA_PATH = PATH + os.path.sep + 'Code/Data/Vegetable Images'
CODE_PATH = PATH + os.path.sep + 'Code'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
CHANNEL = 3
SIZE = 224 # height and width
n_classes = 15

SAVE_FIGURES = False #Says we want to save graphs of the accuracy and loss of the training loop

# create dataframes for train, val, and test data:
train_dir = DATA_PATH + '/train'
val_dir = DATA_PATH + '/validation'
test_dir = DATA_PATH + '/test'


# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 64
DROPOUT = 0.25

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

# Get breakdown of training data distribution
print("Breakdown of Testing Data")
print(train_df.groupby('label_string')['image'].nunique())
print("Breakdown of Validation Data")
print(val_df.groupby('label_string')['image'].nunique())
print("Breakdown of Test Data")
print(test_df.groupby('label_string')['image'].nunique())



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
    plt.clf()

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
    plt.clf()

    # Save Tomato Image
    image = mpimg.imread(train_dir + "/Tomato/0009.jpg")
    plt.title("Tomato Image")
    plt.imshow(image)
    plt.show()
    plt.savefig('Tomato_Image.png', bbox_inches = 'tight')
    plt.clf()


