import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os

from pathlib import Path
import matplotlib.pyplot as plt

# reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

class FSC147_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, density_map_dir, img_dir, transform=None, target_transform=None):
        self.filenames = pd.read_csv(csv_file)
        self.density_map_dir = density_map_dir
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # gets the directory of the image from the csv file (img_labels) and joins it with the image directory
        img_path = os.path.join(self.img_dir, self.filenames.iloc[idx, 1]) # image filename in 1st column
        density_map_path = os.path.join(self.density_map_dir, self.filenames.iloc[idx, 2]) # density map filenam in 2nd column
        
        image = read_image(img_path)
        density_map = torch.from_numpy(np.load(density_map_path))

        # apply any transformations that are specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            density_map = self.target_transform(density_map)
        
        # handle images that are 1 channel (greyscale) by converting to RGB format
        if image.size()[0] == 1:
            # ref https://stackoverflow.com/questions/71957324/is-there-a-pytorch-transform-to-go-from-1-channel-data-to-3-channels
            image = torch.cat([image, image, image], dim=0)

        return image, density_map
    

if __name__ == "__main__":
    img_dir = Path("data\FSC147_384_V2\images_384_VarV2")
    density_map_dir = Path("data\FSC147_384_V2\gt_density_map_adaptive_384_VarV2")
    csv_file = Path("data\dataset.csv")
    
    transform = transforms.Compose([
        transforms.CenterCrop(384), # cropped to be 384 by 384 images
    ])

    target_transform = transforms.Compose([
        transforms.CenterCrop(384), # cropped to be 384 by 384 images
    ])
    
    train_data = FSC147_Dataset(csv_file, density_map_dir, img_dir, transform=transform, target_transform=target_transform) 

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # get the first batch of images/labels
    # in our training, this is where we would actually loop over data and feed into network
    train_images, train_dmaps = next(iter(train_loader))

    # first img/dmap from batch
    img = train_images[0].squeeze()
    dmap = train_dmaps[0].squeeze()

    f, axarr = plt.subplots(1,2)
    # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch 
    # ref: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    axarr[0].imshow(img.permute(1,2,0))
    axarr[1].imshow(dmap, cmap="gray")

    plt.show()


    




