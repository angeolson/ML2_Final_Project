# set global vars
PATH = os.getcwd() + '/Code/Data/Vegetable Images'

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

# from inside class:
label = self.data_frame.iloc[idx]['label_string']
img_name = self.data_frame.iloc[idx]['image']
img_path = os.path.join(self.root_dir, label, img_name)

getBytes = transforms.ToTensor()
        imgTensor = getBytes(image)
        R_mean, G_mean, B_mean = (torch.mean(imgTensor, dim = [1,2])).numpy()
        R_std, G_std, B_std = (torch.std(imgTensor, dim = [1,2])).numpy()

        if self.transform:
            norm = transforms.Normalize([R_mean, G_mean, B_mean], [R_std, G_std, B_std])
            image = norm(image)

