import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os

import json
from pathlib import Path

from debug import display_sample

class FSC147_Dataset(torch.utils.data.Dataset):
    """
    The FSC-147 dataset, implemented as a custom pytorch dataset. Always created using the "Dataset_Creator" class.
    The tutorial https://pytorch.org/tutorials/beginner/basics/data_tutorial.html was used as a reference.
    """
    def __init__(self, csv_file, density_map_dir, img_dir, annotation_data, transform=None, target_transform=None):
        """
        The initializer for the class

        :param pathlib path csv_file: The path for the dataset csv file, which has columns for the image filenames and their 
        corresponding density map filenames
        :param pathlib path density_map_dir: The base path for the density map files
        :param pathlib path img_dir: The base path for the image files
        :param dict annotation_data: The loaded json data for the bbox annotations
        :param pytorch transfrom transform: any pytorch transforms to apply to the images
        :param pytorch transfrom target_transform: any pytorch transforms to apply to the density maps
        """
        self.filenames = pd.read_csv(csv_file)
        self.density_map_dir = density_map_dir
        self.img_dir = img_dir
        self.annotation_data = annotation_data
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.filenames)
    
    def __getitem__(self, idx):
        """
        Return one image, density_map, and list of bounding boxes for the corresponding dataset index

        :param int idx: the index of the item to get from the dataset
        """
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

        bboxes = self._get_object_bboxes(img_name)
        return image, density_map, bboxes
    
    def _get_object_bboxes(self, img_name):
        """
        Extracts exemplar bounding boxes in (x1, y1, x2, y2) format as a pytorch tensor from annotation data.

        :param img_name: the filename for the image to get the bounding boxes for
        """
        coordinates = self.annotation_data.get(img_name).get("box_examples_coordinates")
        boxes = []

        for bbox in coordinates:
            
            # only need the top left and bottom right points
            x1, y1 = int(bbox[0][0]), int(bbox[0][1]) 
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            boxes.append([x1, y1, x2, y2])
        
        if len(boxes) > 3:
            boxes = boxes[0:3]

        return torch.tensor(boxes, dtype=torch.float32)

# Create a training, test, or validation version of the FSC147 Dataset
class Dataset_Creator():
    """
    Creates instances of the FSC147_Dataset class for the training, test, or validation data
    depending on the method called.
    """
    @staticmethod
    def get_training_dataset(transform=None):
        """
        Get an instance of a FSC147_Dataset class that contains all of the training data

        :param pytorch transform transform: any pytorch transform to apply to the images
        """
        csv_file = Path("../data/csv/train_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    @staticmethod
    def get_test_dataset(transform=None):
        """
        Get an instance of a FSC147_Dataset class that contains all of the test data

        :param pytorch transform transform: any pytorch transform to apply to the images
        """
        csv_file = Path("../data/csv/test_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    @staticmethod
    def get_val_dataset(transform=None):
        """
        Get an instance of a FSC147_Dataset class that contains all of the validation data

        :param pytorch transform transform: any pytorch transform to apply to the images
        """
        csv_file = Path("../data/csv/val_dataset.csv")
        return Dataset_Creator._get_dataset(csv_file, transform)

    @staticmethod
    def _get_dataset(csv_file, transform):
        """
        Return an instance of a FSC147_Dataset class using the given csv file path

        :param pathlib path csv_file: Path to the csv file for the dataset
        """
        img_dir = Path("../data/FSC147_384_V2/images_384_VarV2")
        density_map_dir = Path("../data/FSC147_384_V2/gt_density_map_adaptive_384_VarV2")

        # parse json for the example bounding boxes
        with open(Path("../data/annotation_FSC147_384.json")) as annotation_json:
            annotation_data = json.load(annotation_json)

        return FSC147_Dataset(csv_file, density_map_dir, img_dir, annotation_data, transform=transform, target_transform=None) 

if __name__ == "__main__":
    train_data = Dataset_Creator.get_training_dataset()

    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_bboxes = next(iter(train_loader))

    display_sample(train_images, train_dmaps, train_bboxes[0])