 transform=transforms.Compose([
        transforms.RandomResizedCrop((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
)
    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)
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

    transform=transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
)
for param in transformer.layer4.parameters():
    param.requires_grad = True               
image, label = test[0]
    _[0].savefig('IG.png', bbox_inches = 'tight')
elif option == 'noise':
    attributions_ig_nt = noise_tunnel.attribute(input, nt_samples=4, nt_type='smoothgrad_sq', target=pred_label_idx)
elif option == 'shap':