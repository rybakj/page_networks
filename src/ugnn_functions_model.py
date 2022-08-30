from pathlib import Path
import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
from torch._C import parse_schema
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_add_pool
import torch.nn as nn

import networkx as nx
from pathlib import Path

from scipy.sparse import csr_matrix

import os
import pickle
import numpy as np
import pandas as pd
import random
from itertools import chain
import src.randomwalks as rw
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from scipy.sparse import csr_matrix

import os
import pickle
import numpy as np

import random
from gensim.models import Word2Vec

from src.functions import Graph

import matplotlib.pyplot as plt

class GCN_edgeweight(torch.nn.Module):
    def __init__(self, hidden_channels, encoding_dim, input_dim):
        super().__init__()
        torch.manual_seed(1234567)
        # self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv1 = GCNConv(input_dim, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], hidden_channels[1])
        # self.classifier = Linear(2, dataset.num_classes)
        self.linear = Linear(hidden_channels[1], encoding_dim)

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight )
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = x.relu()
        h = self.linear(x)
        return h

def train_epoch(model, data, optimizer, criterion):
    '''
    Single training epoch
    
    Inputs:
    - model: pytorch geometric graph neural network model
    - data: pytorch geometric dataset. needs to contain attributes: "weight", "edge_index", "node_feature", "train_mask", "val_mask", "negative_matrix" and "positive_matrix"
    - optimizer: pytorch optimizer
    - criterion: loss function (taking arguments model, data.positive_matrix, data.negative_matrix, data.train_mask)
    
    Outputs:
    - loss
    - encoding of nodes (h)
    - validation loss
    '''
    optimizer.zero_grad()  # Clear gradients.
    h = model(data.node_feature, data.edge_index, edge_weight = data.weight)

    loss = criterion( h, data.positive_matrix, data.negative_matrix, data.train_mask)
    val_loss = criterion( h, data.positive_matrix, data.negative_matrix, data.val_mask)

    print(loss)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss, h, val_loss

def train(model, data, optimizer, criterion, num_epochs, dir_path, patience = 20, model_name = None):
    '''
    Inputs:
    - model: pytorch geometric graph neural network model
    - data: pytorch geometric dataset. needs to contain attributes: "weight", "edge_index", "node_feature", "train_mask", "val_mask", "negative_matrix" and "positive_matrix"
    - optimizer: pytorch optimizer
    - criterion: loss function (taking arguments model, data.positive_matrix, data.negative_matrix, data.train_mask)
    - num_epochs: number of epochs
    - dir_paths: path to save model checkpoints
    - patience: patience for early stopping
    - model_name: name of model checkpoints (if None, use name "model").
    
    Outputs:
    - train loss history
    - validation loss history
    - node encodings (h)
    '''

    best_val_loss = np.infty

    loss_hist_train = list()
    accuracy_hist_train = list()
    loss_hist_val = list()
    accuracy_hist_valid = list()


    cnt_wait = 0

    for epoch in range(num_epochs):

        if model_name == None:
            model_name = "model"
        else:
            pass
        
        loss, h, val_loss = train_epoch(model, data, optimizer, criterion )
        print(val_loss)
        
        
        if val_loss < best_val_loss:
          best_val_loss = val_loss
          print("Saving at epoch", epoch)


          path_save = dir_path + "/" +  model_name
          torch.save(model.state_dict(), path_save)

          cnt_wait = 0

        else:
            cnt_wait += 1

        if cnt_wait == patience:
            print('Early stopping!')
            break


        h = model(data.node_feature, data.edge_index, data.weight)

        loss_hist_train.append(loss.item())
        loss_hist_val.append(val_loss.item())

         
        print("Epoch:", epoch, "loss: ", loss.item()) 
      
    return(loss_hist_train,  loss_hist_val, h)





class neg_sampling_loss(nn.Module):
    '''
    Loss function. Uses negative sampling.
    '''
    
    def __init__(self, weight=None, size_average=True):
        super(neg_sampling_loss, self).__init__()

    def forward(self, embeddings, neighbors_array, negative_array, mask_array):
         '''
         Inputs:
         - embeddings: matrix of node embeddings (obtained by passing data through GNN model)
         - neighbors_array: matrix of positive samples for each node. 
                     Row corresponds to a source node. In columns are indices of positive samples.
         - negative_array: matrix of negative samples for each node. Same structure as "neighbors_array"
         - mask_array: mask (array of zeros and ones)
         
         '''

        loss = 0

        for i in np.arange(embeddings.shape[0]):
          
          if mask_array[i] == 1:
            neighbors = neighbors_array[i]
            neighbors = neighbors[~np.isnan(neighbors_array[i])]

            pos_products = torch.matmul(embeddings[neighbors], embeddings[i])
            neg_products = torch.matmul(embeddings[negative_array[i]], embeddings[i])

            EPS = 1e-15
            pos_loss = -torch.log(torch.sigmoid(pos_products) + EPS).mean()
            neg_loss = -torch.log(1 - torch.sigmoid(neg_products) + EPS).mean()
            
            if np.isnan((pos_loss + neg_loss).item()): 
              pass
            else:
              # if mask_array[i]:
              loss += pos_loss + neg_loss
              # else:
                # pass
          else:
            pass

        return loss / np.sum(mask_array)



def plot_losses(history):
    
    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 2, 1)
    plt.plot(history[0], lw=4)
    # plt.plot(history[1], lw=4)
    plt.legend(['Train loss'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)

    # ax = fig.add_subplot(1, 2, 2)
    plt.plot(history[1], lw=4)
    # plt.plot(history[2], lw=4)
    plt.legend(['Train acc.', 'Validation acc.'], fontsize=15)
    ax.set_xlabel('Epochs', size=15)
 
    return()