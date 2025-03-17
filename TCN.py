import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

import torch.nn.functional as F
from torch import nn

class TCN(nn.Module):

    def __init__(self, input_size, output_size, num_channels, 
                 kernel_size=2, dropout=0.3, time=100):
        super(TCN, self).__init__()
        self.input_size=input_size
        self.tcn = self.create_temporal_blocks(input_size, num_channels, kernel_size, dropout=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_channels[-1], output_size)  
        self.tempmaxpool = nn.MaxPool1d(time)

    def create_temporal_blocks(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        blocks = []
        for i in range(len(num_channels)):
            dilation_size = 2 ** i

            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            blocks.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout))

        return nn.Sequential(*blocks)


    def forward(self, x):
        output = self.tcn(x.transpose(1, 2))
        output = self.tempmaxpool(output).squeeze(-1)
        out = self.fc(output)
        # 注释掉下面的softmax
        # out = F.softmax(out, dim=1)
        return out


   
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
#        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
#        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.dropout1,
                                 self.conv2, self.chomp2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
