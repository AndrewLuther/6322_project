import os
from pathlib import Path
import pandas as pd
import json

def make_CSV(image_names, csv_name):
    """
    image_names: list of image filenames for the dataset split \n
    csv_name: name of the output csv file
    """
    density_map_names = [name.replace("jpg", "npy") for name in image_names]

    df = pd.DataFrame(data={'jpg_name':image_names, 'dmap_name': density_map_names})
    df.to_csv(Path("data\csv") / Path(csv_name))

if __name__ == "__main__":
    with open(Path("data\Train_Test_Val_FSC_147.json")) as train_test_json:
        train_test = json.load(train_test_json)
        test_names = train_test.get("test")
        train_names = train_test.get("train")
        val_names = train_test.get("val")
        test_coco_names = train_test.get("test_coco")
        val_coco_names = train_test.get("val_coco")
    
    make_CSV(test_names, "test_dataset.csv")
    make_CSV(train_names, "train_dataset.csv")
    make_CSV(val_names, "val_dataset.csv")
