import random_walk.utils.randomwalks as rw
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path

from scipy.sparse import csr_matrix

import os
import pickle
import numpy as np





def get_er_graph(gfile):
    '''Return graph of economic recovery pages, based on a given gpickle file'''

    G = nx.read_gpickle(gfile)  # .to_undirected()

    # reformat the graph to make it compliant with existing random walk functions
    # i.e. add the path to a name property and set the index to be a number

    for index, data in G.nodes(data=True):
        data['properties'] = dict()
        data['properties']['name'] = index

    G2 = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    # get adjacency matrix of G
    A = nx.adj_matrix(G, weight=None)

    T = get_transition_matrix(G)

    return (G, A, T)


def get_transition_matrix(G):
    '''For a given graph G, return a directed transition matrix. The transition probability of moving from
    page A to page B is based on the "edgeweights", i.e. on the number of distinct sessions that move
    between Page A and Page B.'''

    # Create array with edge weight
    T = nx.adjacency_matrix(G, weight="edgeWeight").todense()
    T_array = np.array(T)

    # Transform edge weight into probabilities

    # Normalisation so that probs for each row sum to 1
    sum_of_rows = T_array.sum(axis=1)
    T_probs = T_array / sum_of_rows[:, np.newaxis]  # increase dimension for broadcasting

    # Rows with only 0s = nan. Replace nan values with 1/Tarray.shape[0]
    np.nan_to_num(T_probs, nan=1 / T_array.shape[0], copy=False)

    # check probs sum to one across row:
    print("Max row sum of probs:", np.round(T_probs.sum(axis=1).max(), 4))
    print("Max row sum of probs:", np.round(T_probs.sum(axis=1).min(), 4))

    # Convert into a transition matrix (for random walks function)
    T_directed = csr_matrix(T_probs)

    # Explanation taken from help of "create_networks_graph" in "src/utils/create_functional_network.py":
    # "The edge weight `edgeWeight` is the number of
    #    distinct sessions that move between Page A and Page B.  Created with the
    #    function `extract_nodes_and_edges()`""

    return (T_directed)

def load_pickle_file(file_name, dir_path="./data/outputs"):
    file_path = Path(dir_path + "/" + file_name)
    f = open(file_path, 'rb')
    file = pickle.load(f)
    f.close()

    return (file)


def dump_pickle_file(file, file_name, dir_path="./data/outputs"):
    # check directory exists (if not, create it)
    isdir = os.path.isdir(dir_path)

    if isdir == False:
        os.mkdir(dir_path)
        print("Directory created.")
    else:
        pass

    file_path = os.path.join(dir_path, file_name)

    f = open(file_path, 'wb')
    pickle.dump(file, f)

    return ()

