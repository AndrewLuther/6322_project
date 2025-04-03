from util import Util

import matplotlib.pyplot as plt
import torch

def display_sample(train_images, train_dmaps, train_bboxes):
    """
    Used to ensure dataset loading is working properly. Displays the given image, its density map, and one exemplar.
    """

    # first img/dmap from batch
    img = train_images[0].squeeze().to(torch.int)
    dmap = train_dmaps[0].squeeze()

    # get one example object from image
    examples = Util.get_examples_from_bboxes(img, train_bboxes)
    example = examples[0].squeeze().to(torch.int)

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


def display_prediction(train_images, train_dmaps, pred_dmaps):
    """
    Used for debugging. Displays a given training image, it's ground truth d_map, and a predicted d_map.
    """

    # first img/dmap from batch
    img = train_images[0].squeeze().to(torch.int)
    dmap = train_dmaps[0].squeeze().detach().cpu().numpy()
    pred_dmap = pred_dmaps[0].squeeze().detach().cpu().numpy()

    # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch 
    # ref: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    f, axarr = plt.subplots(1, 3, figsize=(12, 4))
    axarr[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    axarr[0].set_title("Original Image")
    axarr[1].imshow(dmap, cmap="gray")
    axarr[1].set_title("Density Map")
    axarr[2].imshow(pred_dmap, cmap="gray")
    axarr[2].set_title("Prediction Density Map")

    plt.show()


def save_prediction(train_images, train_dmaps, pred_dmaps, filepath):
    """
    Used for debugging. Saves a given training image, it's ground truth d_map, and a predicted d_map to the specified
    filepath.
    """

    # first img/dmap from batch
    img = train_images[0].squeeze().to(torch.int)
    dmap = train_dmaps[0].squeeze().detach().cpu().numpy()
    pred_dmap = pred_dmaps[0].squeeze().detach().cpu().numpy()

    # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch 
    # ref: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    f, axarr = plt.subplots(1, 3, figsize=(12, 4))
    axarr[0].imshow(img.permute(1, 2, 0).detach().cpu().numpy())
    axarr[0].set_title("Original Image")
    axarr[1].imshow(dmap, cmap="gray")
    axarr[1].set_title("Density Map")
    axarr[2].imshow(pred_dmap, cmap="gray")
    axarr[2].set_title("Prediction Density Map")

    plt.savefig(filepath)


def save_image(tensor, filepath, three_dim=False, tensor2=None):
    """
    Used for debugging. Saves an image to the given filepath for the given tensor. If it's three dimensional,
    must set three_dim to True. Can also optionally include a second 2d tensor to display.
    """

    if three_dim:
        tensor = tensor/(torch.max(tensor))
        tensor = tensor.permute(1,2,0)
    tensor = tensor.squeeze().detach().cpu().numpy() # squeeze removes any dimensions with 1

    # ref: https://stackoverflow.com/questions/53623472/how-do-i-display-a-single-image-in-pytorch 
    # ref: https://stackoverflow.com/questions/41793931/plotting-images-side-by-side-using-matplotlib
    f, axarr = plt.subplots(1, 3, figsize=(12, 4))

    if three_dim:
        axarr[0].imshow(tensor)
        axarr[0].set_title("Image")
    else:
        axarr[0].imshow(tensor, cmap="gray")

    if tensor2 != None:
            tensor2 = tensor2.squeeze().detach().cpu().numpy()
            axarr[1].imshow(tensor2, cmap="gray")

    plt.savefig(filepath)
    