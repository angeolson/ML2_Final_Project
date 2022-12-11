train = ImagesDataset(
    data_frame=train_df,
    root_dir=train_dir,
    transform=transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)

validation = ImagesDataset(
    data_frame=val_df,
    root_dir=val_dir,
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)

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

for param in transformer.parameters():
    param.requires_grad = False
for param in transformer.layer4.parameters():
    param.requires_grad = True

                               
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