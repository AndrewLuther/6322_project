import torch

from data import Dataset_Creator, display_sample, display_prediction
from network import FamNet

def train_FamNet():

    # Testing code
    train_data = Dataset_Creator.get_training_dataset()

    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_examples, train_bboxes = next(iter(train_loader))

    # TO DO: this should be done already beforehand
    # get rid of channel dimension, not needed for bboxes
    train_bboxes = train_bboxes[:, 0, :] 

    # # display_sample(train_images, train_dmaps, train_examples)

    model = FamNet()
    model.train(False)

    pred = model(train_images, train_bboxes)
    display_prediction(train_images, train_dmaps, pred)



if __name__ == "__main__":
    train_FamNet()