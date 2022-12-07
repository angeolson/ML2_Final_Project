# purpose: Iterates on the basic-model.py model


# imports
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import argparse
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import torchvision
from torchvision import models
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

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
option = 'noise'

# %% ----------------------------------- Hyper Parameters --------------------------------------------------------------
BATCH_SIZE = 32
DROPOUT = 0.25

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

# %% -------------------------------------- Interpretation: SHAP ----------------------------------------------------------
# https://shap.readthedocs.io/en/latest/example_notebooks/image_examples/image_classification/PyTorch%20Deep%20Explainer%20MNIST%20example.html
# import shap
# batch = next(iter(test_loader))
# images, labels = batch
#
# background = images[:29].to(device)
# test_images = images[29:32].to(device)
#
# torch.cuda.empty_cache()
# e = shap.DeepExplainer(model, background)
# shap_values = e.shap_values(test_images)
#
# shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
# test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
# shap.image_plot(shap_numpy, -test_numpy)

# %% -------------------------------------- Interpretation: Integrated Gradients ----------------------------------------------------------
# map label indexes back to a class name
classes_dict = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}

# plot an example of an image
image, label = test[0]
input = image.unsqueeze(0).to(device)
output = model(input)
prediction_score, pred_label_idx = torch.topk(output, 1) # gets prediction and index of prediction
pred_label_idx.squeeze() # changes size
label_int = pred_label_idx.squeeze().cpu().numpy().item()
label_string = classes_dict[label_int]
default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                     [(0, '#ffffff'),
                                                      (0.25, '#000000'),
                                                      (1, '#000000')], N=224)
if option == 'ig':
    # Option I: integrated Gradients
    integrated_gradients = IntegratedGradients(model)
    attributions_ig = integrated_gradients.attribute(input, target=pred_label_idx, n_steps=100)



    _ = viz.visualize_image_attr(np.transpose(attributions_ig.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                 method='heat_map',
                                 cmap=default_cmap,
                                 show_colorbar=True,
                                 sign='positive',
                                 outlier_perc=1)
    _[0].savefig('IG.png', bbox_inches = 'tight')

elif option == 'noise':
# Option 2: Integrated Gradients with noise tunnel
    integrated_gradients = IntegratedGradients(model)
    noise_tunnel = NoiseTunnel(integrated_gradients)

    attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=4, nt_type='smoothgrad_sq', target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "positive"],
                                          cmap=default_cmap,
                                          show_colorbar=True)
    _[0].savefig('IG_noise.png', bbox_inches = 'tight')

elif option == 'shap':
    # Option 3: GradientShap with blank (black) reference image
    torch.manual_seed(0)
    np.random.seed(0)

    gradient_shap = GradientShap(model)

    # Defining baseline distribution of images
    rand_img_dist = torch.cat([input * 0, input * 1])

    attributions_gs = gradient_shap.attribute(input,
                                              n_samples=50,
                                              stdevs=0.0001,
                                              baselines=rand_img_dist,
                                              target=pred_label_idx)
    _ = viz.visualize_image_attr_multiple(np.transpose(attributions_gs.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          np.transpose(image.squeeze().cpu().detach().numpy(), (1,2,0)),
                                          ["original_image", "heat_map"],
                                          ["all", "absolute_value"],
                                          cmap=default_cmap,
                                          show_colorbar=True)
    _[0].savefig('IG_shap.png', bbox_inches='tight')