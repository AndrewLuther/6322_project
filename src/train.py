import torch

from network import FeatureExtractionModule
from data import Dataset_Creator


def train_FamNet():
    train_data = Dataset_Creator.get_training_dataset()

    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)

    # eventually this will be a loop through the data
    train_images, train_dmaps, train_examples = next(iter(train_loader))

    model = FeatureExtractionModule()
    
    # will be true
    model.train(False)

    # extract features from the image
    image_features1, image_features2 = model(train_images)

    # extract features from first example
    example_features1, example_features2 = model(train_examples[0])

    # how do we correlate between these? They are different sizes
    print(image_features1.size())
    print(example_features1.size())


if __name__ == "__main__":
    train_FamNet()