parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()

url = 'https://drive.google.com/u/3/uc?id=16U5WG2Jo8mU-J2jpo3PCnsvAn2CF2hF0&export=download'
PATH = args.path
# PATH ='/home/ubuntu/Final-Project-Group4'
DATA_PATH = PATH + os.path.sep + 'Data'
os.chdir(DATA_PATH)
gdown.download(url, 'archive(4).zip', quiet=False)
os.system("unzip 'archive(4).zip'")
print('Done!')

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
# PATH = '/home/ubuntu/Final-Project-Group4'
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
export_data = True

BATCH_SIZE = 64
DROPOUT = 0.25

# set to data loaders
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

print("Starting testing loop...")

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

if export_data is True:
    test_df.to_csv('test_predictions.csv')

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
# PATH = '/home/ubuntu/Final-Project-Group4'
DATA_PATH = PATH + os.path.sep + 'Data/Vegetable Images'
CODE_PATH = PATH + os.path.sep + 'Code'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
CHANNEL = 3
SIZE = 224 # height and width
n_classes = 15
model_type = 'transformer' # options: transformer or CNN

SAVE_MODEL = True #Says we want to save the best model during training loop
SAVE_FIGURES = True
LR = 1e-3
N_EPOCHS = 30
BATCH_SIZE = 64
DROPOUT = 0.25

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

# create dataframes for train, val, and test data:
train_dir = DATA_PATH + '/train'
val_dir = DATA_PATH + '/validation'

# create splits and encode:
train_df = getFrame(train_dir).reset_index(drop=True)
val_df = getFrame(val_dir).reset_index(drop=True)


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

if model_type == 'CNN':
    model = CNN().to(device)
else:
    model = transformer.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=LR)
criterion = nn.BCELoss()

os.chdir(CODE_PATH)
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
        preds = output.detach().cpu().numpy()
        new_preds = np.zeros(preds.shape)
        for i in range(len(preds)):
            new_preds[i][np.argmax(preds[i])] = 1
        accuracy = accuracy_score(y_true=labels.cpu().numpy().astype(int),
                                      y_pred=new_preds.astype(int))
        train_acc.append(accuracy)
        optimizer.step()

    val_losses = []
    val_acc = []
    model.eval()
    for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        val_loss = criterion(output, labels.float())
        val_losses.append(val_loss.item())
        preds = output.detach().cpu().numpy()
        new_preds = np.zeros(preds.shape)
        for i in range(len(preds)):
            new_preds[i][np.argmax(preds[i])] = 1
        accuracy = accuracy_score(y_true=labels.cpu().numpy().astype(int),
                                      y_pred=new_preds.astype(int))
        val_acc.append(accuracy)

    epoch_train_loss = np.mean(train_losses)
    epoch_val_loss = np.mean(val_losses)
    epoch_train_acc = np.mean(train_acc)
    epoch_val_acc = np.mean(val_acc)
    epoch_tr_loss.append(epoch_train_loss)
    epoch_vl_loss.append(epoch_val_loss)
    epoch_tr_acc.append(epoch_train_acc)
    epoch_vl_acc.append(epoch_val_acc)
    #prints the epoch, the training and validation accuracy and loss
    print(f'Epoch {epoch + 1}')
    print(f'train_loss : {epoch_train_loss} val_loss : {epoch_val_loss}')
    print(f'train_accuracy : {epoch_train_acc} val_accuracy : {epoch_val_acc}')

parser = argparse.ArgumentParser()
parser.add_argument("--path", default=None, type=str, required=True)  # Path of file
args = parser.parse_args()
PATH = args.path
# PATH = '/home/ubuntu/Final-Project-Group4'
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

BATCH_SIZE = 32
DROPOUT = 0.25

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

# set to data loaders
test_loader = DataLoader(test, batch_size=BATCH_SIZE)

transformer = models.resnet34(pretrained=True)


transformer.fc = nn.Sequential(
    nn.Linear(transformer.fc.in_features, n_classes),
    nn.Softmax(dim=1)
                               )

if model_type == 'CNN':
    model = CNN().to(device)
else:
    model = transformer.to(device)

model.load_state_dict(torch.load('model_main.pt', map_location=device))
model.to(device)

criterion = nn.BCELoss()

# map label indexes back to a class name
classes_dict = {0: 'Bean', 1: 'Bitter_Gourd', 2: 'Bottle_Gourd', 3: 'Brinjal', 4: 'Broccoli', 5: 'Cabbage', 6: 'Capsicum', 7: 'Carrot', 8: 'Cauliflower', 9: 'Cucumber', 10: 'Papaya', 11: 'Potato', 12: 'Pumpkin', 13: 'Radish', 14: 'Tomato'}
label_string = classes_dict[label_int]
    _[0].savefig('IG.png', bbox_inches = 'tight')

    _[0].savefig('IG_noise.png', bbox_inches = 'tight')

    _[0].savefig('IG_shap.png', bbox_inches='tight')