import torch
from data import Dataset_Creator, display_sample
import torchvision.transforms as transforms

# The feature extraction module in the paper is ResNet50

if __name__ == "__main__":

    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = Dataset_Creator.get_training_dataset(transform=preprocess)
    # in paper a batch size of 1 is specified
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=True)
    train_images, train_dmaps, train_examples = next(iter(train_loader))

    display_sample(train_images, train_dmaps, train_examples)

    # ref: https://pytorch.org/hub/pytorch_vision_resnet/
    # I think the weights are loaded here automatically? https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50')

    model.train(False)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        train_images = train_images.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(train_images)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Read the categories
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    # Show top categories per image
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print(categories[top5_catid[i]], top5_prob[i].item())
