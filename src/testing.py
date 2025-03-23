import torch

from data import Dataset_Creator
from util import Util


# for testing out functionality from other files

if __name__ == "__main__":
    train_data = Dataset_Creator.get_training_dataset()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
    train_images, train_dmaps, train_examples, train_bboxes = next(iter(train_loader))

    examples = Util.get_examples_from_bboxes(train_images[0], train_bboxes[0])

    # test with same dmap
    print(Util.min_count_loss(train_dmaps, train_dmaps))