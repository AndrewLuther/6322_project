import numpy as np
import torch

# This file is for any misc data processing functions needed for altering the data inputs/outputs

class Util():
    def calculate_correlation(a, b):
        # ref https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
        a = torch.tensor([[1,2,3],
                    [4,5,6]]).detach().numpy()

        b = torch.tensor([[-1,2,3],
                    [4,-6,6]]).detach().numpy()

        # I'm not really sure if this is the right thing, but it's a correlation that takes
        # 2 tensors as input and outputs another tensor? 

        # *** a and b need to be the same size
        correlation = torch.from_numpy(np.corrcoef(a, b))
        return correlation