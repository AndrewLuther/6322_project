import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import pandas as pd
import os

from pathlib import Path
import matplotlib.pyplot as plt

# reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

label_map = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "casette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute"
}

class ImageNetteDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        # gets the directory of the image from the csv file (img_labels) and joins it with the image directory
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]
        image = read_image(img_path)

        # apply any transformations that are specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        
        return image, label
    

if __name__ == "__main__":
    datasetPath = Path("imagenette2-160/noisy_imagenette.csv")
    img_dir = Path("imagenette2-160")

    
    transform = transforms.Compose([
        transforms.CenterCrop(160), # cropped to be 160 by 160 images, this will autopad for smaller images
        transforms.Grayscale(1) # convert to a single grayscale channel
    ])
    
    train_data = ImageNetteDataset(datasetPath, img_dir, transform=transform) 

    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # get the first batch of images/labels
    # in our training, this is where we would actually loop over data and feed into network
    train_features, train_labels = next(iter(train_loader))

    # first img/label from batch
    img = train_features[0].squeeze()
    label = label_map.get(train_labels[0]) # translate the label code to the actual label

    # display img
    plt.imshow(img, cmap="gray")
    plt.show()
    print(label)


    




