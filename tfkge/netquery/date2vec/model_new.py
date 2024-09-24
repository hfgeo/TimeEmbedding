import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import torch.utils.data
import math
import numpy as np


class Date2Vec(nn.Module):
    """
        Creates a single layer fully connected feed forward neural network.
        this will use non-linearity, layer normalization, dropout
        this is for the hidden layer, not the last layer of the feed forard NN
    """

    def __init__(self, input_dim,
                 output_dim=64,
                 dropout_rate=0.15,
                 device = 'cuda:0',
                 skip_connection=False):

        super(Date2Vec, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout_rate)
        self.act = torch.cos
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.layernorm = nn.LayerNorm(self.output_dim+self.input_dim)
        self.device = device
        # # the skip connection is only possible, if the input and out dimention is the same
        # if self.input_dim == self.output_dim:
        #     self.skip_connection = skip_connection
        # else:
        #     self.skip_connection = False

        self.linear1 = nn.Linear(self.input_dim, self.output_dim)
        self.linear2 = nn.Linear(self.input_dim + self.output_dim, self.output_dim*2)
        self.linear3 = nn.Linear(self.output_dim*2, self.output_dim)
        self.linear4 = nn.Linear(self.output_dim , self.input_dim)

        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
        nn.init.xavier_uniform_(self.linear4.weight)




    def encoder(self, x):

        out1 = self.linear1(x)
        out2 = self.act(out1)
        out = self.dropout(out2)
        out = torch.cat([out, x], 1)
        out = self.layernorm(out)

        out1 = self.linear2(out)
        out1 = self.act1(out1)
        out = self.dropout(out1)

        return out

    def decoder(self,x):

        out1 = self.linear3(x)
        out1 = self.act1(out1)
        out = self.dropout(out1)

        out1 = self.linear4(out)
        out1 = self.act(out1)
        out = self.dropout(out1)
        return out

    def forward(self, input_tensor):

        encode = self.encoder(input_tensor)
        output = self.decoder(encode)

        return output

