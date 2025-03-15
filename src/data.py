import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os

import json
from pathlib import Path
import matplotlib.pyplot as plt

# reference: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html 

class FSC147_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, density_map_dir, img_dir, annotation_data, transform=None, target_transform=None):
        self.filenames = pd.read_csv(csv_file)
        self.density_map_dir = density_map_dir
        self.img_dir = img_dir
        self.annotation_data = annotation_data
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        # gets the directory of the image from the csv file (img_labels) and joins it with the image directory
        img_name = self.filenames.iloc[idx, 1] # image filename in 1st column
        img_path = os.path.join(self.img_dir, img_name) 
        density_map_path = os.path.join(self.density_map_dir, self.filenames.iloc[idx, 2]) # density map filename in 2nd column
        
        image = read_image(img_path)
        density_map = torch.from_numpy(np.load(density_map_path))

        image = image.to(torch.float)

        # apply any transformations that are specified
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            density_map = self.target_transform(density_map)
        
        # handle images that are 1 channel (greyscale) by converting to RGB format
        if image.size()[0] == 1:
            # ref https://stackoverflow.com/questions/71957324/is-there-a-pytorch-transform-to-go-from-1-channel-data-to-3-channels
            image = torch.cat([image, image, image], dim=0)

        examples = self._get_object_examples(image, img_name)
        return image, density_map, examples
    
    def _get_object_examples(self, image, img_name):
        # get the example bbox annotations
        coordinates = self.annotation_data.get(img_name).get("box_examples_coordinates")
        examples = []
        for bbox in coordinates:
            # crop the image to the bbox, only need top left point and bottom right point
            x1=[int(bbox[0][1]), int(bbox[0][0])]
            x2=[int(bbox[2][1]), int(bbox[2][0])]
            # add 1 because slice operators don't include larger index
            example = image[:, x1[0]:x2[0]+1, x1[1]:x2[1]+1]
            examples.append(example)
        
        return examples

# Create a training, test, or validation version of the FSC147 Dataset
class Dataset_Creator():
    @staticmethod
    def get_training_dataset(transform=None):
        csv_file = Path("../data/csv/train_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    @staticmethod
    def get_test_dataset(transform=None):
        csv_file = Path("../data/csv/test_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    @staticmethod
    def get_val_dataset(transform=None):
        csv_file = Path("../data/csv/val_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    def _get_dataset(csv_file, transform):
        img_dir = Path("../data/FSC147_384_V2/images_384_VarV2")
        density_map_dir = Path("../data/FSC147_384_V2/gt_density_map_adaptive_384_VarV2")

        # parse json for the example bounding boxes
        with open(Path("../data/annotation_FSC147_384.json")) as annotation_json:
            annotation_data = json.load(annotation_json)

        return FSC147_Dataset(csv_file, density_map_dir, img_dir, annotation_data, transform=transform, target_transform=None) 

def display_sample(train_images, train_dmaps, train_examples):
    """
    Display a random sample, its density map, and an example object from the given dataset
    """

    # first img/dmap from batch
    img = train_images[0].squeeze().to(torch.int)
    dmap = train_dmaps[0].squeeze()

    # get one example object from image
    example = train_examples[0][0].squeeze().to(torch.int)
    
    # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch 
    # ref: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    f, axarr = plt.subplots(1, 3, figsize=(12, 4))
    axarr[0].imshow(img.permute(1, 2, 0))
    axarr[0].set_title("Original Image")
    axarr[1].imshow(dmap, cmap="gray")
    axarr[1].set_title("Density Map")
    axarr[2].imshow(example.permute(1, 2, 0))
    axarr[2].set_title("Exemplar Image")

    plt.show()

if __name__ == "__main__":
    train_data = Dataset_Creator.get_training_dataset()

    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_examples = next(iter(train_loader))

    display_sample(train_images, train_dmaps, train_examples)