import torch
import torch.nn as nn

class U_NET(nn.Module):
    def __init__(self, in_channels, out_channels, depth, Max_channels, ):
        super(U_NET, self).__init__()
