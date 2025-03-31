import torch

from torch.utils.tensorboard import SummaryWriter

class Logger():
    def __init__(self):
        self.writer = SummaryWriter()
        self.iteration = 0

    def add_scalar(self, scalar_name, scalar_value):
        self.writer.add_scalar(scalar_name, scalar_value, self.iteration)
    
    def add_histogram(self, hist_name, hist_val):
        self.writer.add_histogram(hist_name, hist_val, self.iteration)

    def increment(self):
        self.iteration += 1

    def finish(self):
        self.writer.flush()
        self.writer.close()

    ####

    def forward_hook(self, module, input, output):
        self.add_histogram("weight", module.weight)

    def backward_hook(self, module, input, output):
        self.add_histogram("grad", module.weight.grad)