import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt



class ContextNN(nn.Module):
    """
    The contextual network of ProMP.
    n_input: the dimension of contextual goal / object.
    n_hidden: the size of hidden layers.
    n_output: the dimension of ProMP weights.
    """
    def __init__(self, n_input, n_hidden, n_output):
        super(ContextNN,self).__init__()
        self.seq = nn.Sequential(nn.Linear(n_input, n_hidden, bias=False),
                                 # TODO: Adjust here
                                 #nn.ReLU(),
                                 #nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_output, bias=False),
                                 nn.Tanh())


    def forward(self, input):
        out = self.seq(input)
        return out

