import torch
import numpy as np
from torch.nn import *

#process data

#build model

class GPT(Module):
    def __init__(self):
        super(GPT, self).__init__()

class MultiHeadSelfAttention(Module):

    def __init__(self, n_embd = 256, n_heads = 8, bigBird = False):
        super(SelfAttention, self).__init__()                       # ???

        self.n_embd = n_embd

        self.W_k = Linear(n_embd, n_embd)
        self.W_q = Linear(n_embd, n_embd)
        self.W_v = Linear(n_embd, n_embd)

    def forward(self, X):

        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)

        #mask Q @ K.T               #TODO move these masks to main class
        if bigBird:
            n_neighbours = 5
            n_memory = 2
            mask = np.zeros((n_embd, n_embd), dtype = bool)
            for i in range(-n_neihbours, n_neighbours + 1):
                mask + = np.diag(n_embd, k = i)
                
        else:
            mask = np.tril(np.ones(n_embd, n_embd))
            mask.astype(bool)

        masked_QK = np.ma.array(Q @ (K.T), mask = mask)

        #calcualte Z
        Z = Softmax(masked_QK / np.sqrt(self.n_embd)) @ V

        #multiply


"""
class SelfAttention(Module):

    def __init__(self, n_embd = 256, bigBird = False):
        super(SelfAttention, self).__init__()           # ???

        self.n_embd = n_embd

        self.W_k = Linear(n_embd, n_embd)
        self.W_q = Linear(n_embd, n_embd)
        self.W_v = Linear(n_embd, n_embd)

    def calculateScore(self, X):

        K = self.W_k(X)
        Q = self.W_q(X)
        V = self.W_v(X)

        #mask Q @ K.T               #TODO move these masks to main class
        if bigBird:
            n_neighbours = 5
            n_memory = 2
            mask = np.zeros((n_embd, n_embd), dtype = bool)
            for i in range(-n_neihbours, n_neighbours + 1):
                mask + = np.diag(n_embd, k = i)
                
        else:
            mask = np.tril(np.ones(n_embd, n_embd))
            mask.astype(bool)

        masked_QK = np.ma.array(Q @ (K.T), mask = mask)

        #calcualte Z
        Z = Softmax(masked_QK / np.sqrt(self.n_embd)) @ V
"""

"""
class MultiHead(Module):

    def __init__(self, n_heads = 8):
        super(MultiHead, self).__init__()               # ???

        
        z_concat = np.zeros((n_embd, n_heads*n_embd))
        for i in range(n_heads):
            
"""        
                

        
    

#train model

