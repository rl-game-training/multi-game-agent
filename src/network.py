import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DQN(nn.Module):

    def __init__(self, w, h, output_len):
        super(DQN, self).__init__()

        self.conv1 = nn.Conv2d(4, 16, kernel_size=2, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
 

        def conv2_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size-1) - 1) // stride + 1
        
        conv_out_w = conv2_size_out(conv2_size_out(w))
        conv_out_h = conv2_size_out(conv2_size_out(h))
        conv_out_size = conv_out_w * conv_out_h * 32
        print(conv_out_h, conv_out_w)
        self.flattened = nn.Linear(conv_out_size, output_len)

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        print(x.view(x.size(0), -1).size())
        print(x.size())
        x = self.flattened(x.view(x.size(0), -1))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == '__main__':

    s = torch.ones(size=(1, 4, 20, 20))
    net = DQN(20, 20, 7)
    print(net)
    print(net(s))