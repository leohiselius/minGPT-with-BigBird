import torch
import numpy as np
from torch.nn import *

#process data

#build model haha

class GPT(Module):
    def __init__(self):
        super(GPT, self).__init__()

class MultiHeadSelfAttention(Module):

    def __init__(self, n_embd = 784, n_heads = 12, big_bird = False):
        super().__init__()                       # ???

        self.n_embd = n_embd
        self.n_heads = n_heads
        self.n_latent = n_embd // n_heads

        assert (n_embd / n_heads == self.n_latent), \
            "n_heads must equally divide n_embd!"

        self.n_words = None  # number of words in sentence
        self.n_data = None  # number of sentences / data matrices in X tensor

        self.W_k = Linear(self.n_latent*n_heads, n_embd) # torch applies the transpose of Linear ( W_k(x) = x @ W_k.T )
        self.W_q = Linear(self.n_latent*n_heads, n_embd)
        self.W_v = Linear(self.n_latent*n_heads, n_embd)

        self.W0 = Linear(self.n_latent, self.n_latent*n_heads) # output projection (combines the 12 heads into 1)

        # mask Q @ K.T. Mask should not be trained by optimizer and should thus be a buffer

        # TODO : koda klart kontruktionen av np-matrisen, kolla om sparse hjälper i autograd
        # TODO : undersök om vi kan implementera BigBirds roll block - lösning inom autograd
        if big_bird:

            # construct mask with numpy
            n_neighbours = 5
            n_memory = 2
            np_mask = np.zeros((n_embd, n_embd), dtype=bool)
            for i in range(-n_neighbours, n_neighbours + 1):
                np_mask += np.diag(n_embd, k=i)

            # convert to 4D tensor
            tensor_mask = torch.tril(torch.from_numpy(np_mask)).view(1, 1, self.n_embd, self.n_embd)

            # convert to sparse
            non_zero_idx = torch.nonzero(tensor_mask).t() #t is transpose function
            non_zero_values = tensor_mask[non_zero_idx[0],non_zero_idx[1]] # extract non zero values
            mask = torch.sparse.FloatTensor(non_zero_idx,non_zero_values, tensor_mask.size())

        else:
            # define upper triangular matrix stored as 4D tensor for compatibility
            mask = torch.tril(torch.ones(self.n_embd, self.n_embd)).view(1, 1, self.n_embd, self.n_embd)

        self.register_buffer("mask", mask)

    def forward(self, X):
        # pytorch linear treats final dim of data as input dim, other dims preserved
        self.n_data, self.n_words, x_embd = X.size() # referred to as B,T,C in original minGPT

        assert (x_embd == self.n_embd),\
            "Embedding dims do not match! Data: " + str(x_embd) + "!= Model: " + str(self.n_embd)

        K = self.W_k(X) # = tensor, where tensor[i,k...] = X[i,j,..., :] @ W_k.T
        Q = self.W_q(X)
        V = self.W_v(X)

        # K,Q,V: [n_data, n_words, n_latent*n_heads]

        # break stacked heads into separate dims
        K = K.view(self.n_words, self.n_data, self.n_heads, self.n_latent)
        Q = Q.view(self.n_words, self.n_data, self.n_heads, self.n_latent)
        V = V.view(self.n_words, self.n_data, self.n_heads, self.n_latent)

        # K,Q,V: [n_data, n_words, n_heads, n_latent]

        # switch dims 1 & 2 to put each K,Q,V - matrix in last two dims
        K = K.transpose(1, 2)
        Q = Q.transpose(1, 2)
        V = V.transpose(1, 2)

        # K,Q,V: [n_data, n_heads, n_words, n_latent]

        # transpose of matrix should now act on final dims
        attention_mat = Q @ K.transpose(-2,-1) * (self.n_latent**(-1/2))

        # attention_mat: [n_data, n_heads, n_words, n_words]

        attention_mat = attention_mat.masked_fill(self.mask[:,:,:self.n_words,:self.n_words] == 0, float('-inf'))
        attention_mat = functional.softmax(attention_mat, dim=-1)

        Z = attention_mat @ V # Z: [n_data, n_heads, n_words, n_latent]

        Z = Z.transpose(1,2).contiguous().view(self.n_data,self.n_words,self.n_latent*self.n_heads)

        return Z


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

