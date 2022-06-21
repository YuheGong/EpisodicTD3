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
        self.seq = nn.Sequential(nn.Linear(n_input, n_hidden),
                                 # TODO: Adjust here
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_hidden),
                                 nn.ReLU(),
                                 nn.Linear(n_hidden, n_output))
        self.hidden1 = nn.Linear(n_input,n_hidden)
        self.hidden2 = nn.Linear(n_hidden,n_hidden)
        self.predict = nn.Linear(n_hidden,n_output)

    def forward(self, input):
        #out = self.hidden1(input)
        #out = F.ReLU(out)
        #out = self.hidden2(out)
        #out = F.ReLU(out)
        #out =self.predict(out)
        out = self.seq(input)
        return out

