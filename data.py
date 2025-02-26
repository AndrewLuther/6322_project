import torch
import torchvision
import torchvision.transforms as transforms

batch_size = 4
train_data = torchvision.datasets.ImageNet(root='./data', train=True, download=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
