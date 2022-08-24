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
from src.functions import calc_median_difference_n2v

def load_and_preprocess_graph(graph_name, dir, p = 1, q = 1, pages_to_remove = None):
  '''
  Load graph object and rerpocess transition probabilities in N2V way.

  Returns N2V graph object.
  '''
  G = load_pickle_file(graph_name, dir_path=dir)

  if pages_to_remove != None:
    G.remove_nodes_from(pages_to_remove)

  G_n2v = Graph(G, is_directed = True, p = p, q = q)
  G_n2v.preprocess_transition_probs()



  return( G_n2v ) 


def draw_neighbors(graph_nodes, G_primary, G_secondary, num_neighbors, walk_length, num_walks_primary, min_neighbors_tolerance, max_num_secondary_rws, dict_node_index):
  '''
  For a list of graph nodes, generate positive examples for negative sampling.

  Goal:
  - Generate at least "min_neighbors_tolerance" and at most "num_neighbors" positive samples for each node.
  - This is done as follows:

    1. Draw "num_walks_primary" number of random walks from graph "G_primary". Positive samples are "num_neighbors" unique nodes visited in these walks (selected randomly
    if more nodes were visited than the required number of positive examples).
    If number of generated samples for some node is less than "num_neighbors", for this node carry out the next step:
    2. While number of positive samples for a node is not equal to "num_neighbors":
      - Run 40 random walks from this node using graph "G_secondary". Walk length increases by 2
      - Do this at most "max_num_secondary_rws" times.
    3. Discard all nodes for which we have less examples than specified by "min_neighbors_tolerance".


  Inputs:
  - graph_nodes: list of graph nodes
  - G_primary: graph object used to draw primary random walks (most of positive samples come from these walks)
  - G_secondary: graph object for secondary RWs
  - num_neighbors: number of positive samples required per node
  - walk_length: length of primary RWs
  - num_walks_primary: number of primary RWs
  - min_neighbors_tolerance: minimum number of positive smaples tolerated
  - max_num_secondary_rws: maximum number of secondary RWs (higher number means we have more positive samples but takes longer time to run)
  - dict_node_index: mapping of node names to indices

  Returns:
  - ngh_dict: dictionary of positive samples. Key = starting node, item = list of visited nodes
  - failed_nodes: list of nodes for which too few positive samples were generated (these are discarded from later analysis).

  '''

  rws_primary = G_primary.simulate_walks_from_seeds(num_walks_primary, walk_length, seeds = graph_nodes)

  failed_nodes = []
  failed_nodes_index = []
  ngh_dict = dict() # list()

  for node in graph_nodes:

    node_index = dict_node_index[node]

    pages_visited_node = [page for path in rws_primary["paths_taken"][node_index] for page in path[1:]]
    pages_visited_node = np.unique(pages_visited_node)


    # try to get a desired number of distinct neighbors
    i = 0 # count number of extra iterations
    while len(pages_visited_node) < num_neighbors:
      # print( node, ": ", len(pages_visited_node) )
      rws_secondary = G_secondary.simulate_walks_from_seeds(num_walks = 40, walk_length = walk_length, seeds = [node])
      rws_secondary["pages_visited"].remove(node)
      rw_nodes_index = np.array([page for page in rws_secondary["pages_visited"]])
      pages_visited_node = np.unique( np.concatenate([pages_visited_node, rw_nodes_index]) )

      i += 1

      if i == int(max_num_secondary_rws + 1):
        break

    # draw neighbors at random if possible
    # those that don't meet threshold are accumulated in "failed_nodes" list
    if len(list(pages_visited_node)) >= min_neighbors_tolerance:
      print( len(list(pages_visited_node)) )
      try:
        ngh_dict[node] = random.sample(list(pages_visited_node), num_neighbors)

      except:
        ngh_dict[node] = list(pages_visited_node) 


      if len(ngh_dict[node]) == 0:
        print("Stoppping due to an empty array")
        break

    else:
      failed_nodes.append(node)
      failed_nodes_index.append(node_index)
      print("Failed for node", node)
      print(len(failed_nodes))
    
  return(ngh_dict, failed_nodes)


def get_neighbor_matrix(G, ngh_dict, failed_nodes, num_neighbors):
  '''
  Construct a neighborhood matrix. Each row represents a neighborhood of a single node. 
  In columns are indices of the node's neighbors.

  Inputs:
  - G: graph
  - ngh_dict: dictionary (output of "draw_neighbors"). Keys = start nodes, items = list of neighbors
  - failed_nodes: nodes to disregard (due to insufficient number of neighbors)
  - num_neighbors: desired number of neighbors

  Returns:
  - Graph (with failed_nodes removed)
  - neighb_matrix (matrix of neighbors). Row = starting node, Columns = neighbors' indices
  '''

  G.remove_nodes_from(failed_nodes)
  

  graph_nodes = list(G.nodes())

  for node in ngh_dict.keys():
    if node not in graph_nodes:
      del ngh_dict[node]
    else:
      pass

  # remove failed nodes from paths starting in other nodes
  for node in ngh_dict.keys():
    ngh_dict[node] = [x for x in ngh_dict[node] if x in graph_nodes] 

  dict_node_index = dict()

  # mapping of nodes to integers/indices
  for node in list(G.nodes()):
    dict_node_index[node] = graph_nodes.index(node)

  # matrix of neigh indices
  neigh_matrix = np.full(( len(graph_nodes ), num_neighbors), np.nan)

  for index, node in enumerate( G.nodes() ):
    # print(node)
    neigh_inices_arr = np.array([dict_node_index[x] for x in ngh_dict[node]])
    try:
      neigh_matrix[index, :] = neigh_inices_arr
    except:
      neigh_matrix[index, :] = np.concatenate([ neigh_inices_arr, np.full(num_neighbors - len(ngh_dict[node]), np.nan)])

  return( G, neigh_matrix )


from torch_geometric.utils.convert import from_networkx
import pandas as pd

def get_pyg_graph(G, train_set_fraction = 0.9):
  '''
  Format graph so that it's suitable for torch_geometric module.

  Inputs:
  - G: networkx graph
  - train_set_fraction: fraction of train samples

  Outputs:
  - torch geometric graph. Attributes are:
      - train_mask (np.array): 0 for val nodes, 1 for train nodes
      - val_mask (np.array)
      - node_feature: matrix of node features (torch.float32 format)
  '''
  # convert to torch_geometric graph object
  pyg_graph = from_networkx(G)


  data_set_size = len( list(G.nodes()) )


  train_set_size = np.floor(train_set_fraction * data_set_size)
  train_mask = np.zeros( int(data_set_size) )

  train_set =   random.sample( list(np.arange( data_set_size )), int(train_set_size) )
  train_mask[train_set] = 1
  val_mask = 1 - train_mask

  pyg_graph.train_mask = np.array(train_mask) == 1
  pyg_graph.val_mask = np.array(val_mask) == 1

  pyg_graph.node_feature = pyg_graph.node_feature.to(torch.float32)

  return( pyg_graph )



def get_negative_examples(G, neigh_matrix):
  '''
  For each node of graph G, draw negative samples.

  For any given node,s negative samples are randomly selected graph nodes, with positive samples exlcuded.

  Inputs:
  - G: graph
  - neigh_matrix: matirx of positive sample indices

  Returns:
  - matrix of negative samples
  '''

  indices_nodes = np.arange(len(G.nodes()))


  neg_indices = list()

  for row in np.arange(neigh_matrix.shape[0]):
    print(row)
    pi_i = neigh_matrix[row, :]
    pi_i = pi_i[~np.isnan(pi_i)]
    remaining_indices_i = np.delete( indices_nodes, pi_i.astype(int))
    neg_indices_i = random.choices( remaining_indices_i, k = 30 ) 
    neg_indices.append(neg_indices_i)

  negative_array = np.array(neg_indices)

  return(negative_array)




def get_pyg_object(graph_wfeatures_name, graph_orig_name, dir, walk_params_dict, pages_to_remove = None):

  '''
  Construct a torch_geometric-compatible graph object, along with matrices of positive and negative samples.

  Inputs:
  - graph_wfeatures_name: graph containing node features
  - graph_orig_name: original graph object
  - dir: path to the graph objects
  - p_primary, q_primary, p_secondary, q_secondary: parameters of primary and secondary biased random walks
  - num_neighbors: number of desired neighbors
  - walk_length: length of primary RWs
  - num_walks_primary: number of primary RWs
  - min_neighbors_tolerance: minimum number of positive smaples tolerated
  - max_num_secondary_rws: maximum number of secondary RWs (higher number means we have more positive samples but takes longer time to run)
  - pages_to_remove: list of any extra nodes to remove (e.g. if NER analysis shows some pages no longer exist)

  Returns:
  - pyg_graph: torch_geometric graph object
  - neigh_matrix: matrix of positive samples
  - negative_array: matrix of negative samples
  - node_list
  '''


  G3 = load_pickle_file(graph_wfeatures_name, dir_path=dir)

  if pages_to_remove != None:
    G3.remove_nodes_from(pages_to_remove)


  graph_nodes = list(G3.nodes())

  dict_node_index = dict()

  # mapping of nodes to integers/indices
  for node in graph_nodes:
    dict_node_index[node] = graph_nodes.index(node)

  G_n2v_weighted = load_and_preprocess_graph(graph_orig_name, dir = dir, 
                         p = walk_params_dict["walk_pq_primary"]["p"], q = walk_params_dict["walk_pq_primary"]["q"], 
                         pages_to_remove = pages_to_remove)
  G_n2v_standard = load_and_preprocess_graph(graph_orig_name, dir = dir, 
                         p = walk_params_dict["walk_pq_secondary"]["p"], q = walk_params_dict["walk_pq_secondary"]["q"],
                         pages_to_remove = pages_to_remove)

  ngh_list, failed_nodes = draw_neighbors(
                  graph_nodes, G_n2v_weighted, G_n2v_standard, 
                  walk_params_dict["num_neighbors"], walk_params_dict["walk_length"], walk_params_dict["num_walks_primary"], 
                  walk_params_dict["min_neighbors_tolerance"], 
                  walk_params_dict["max_num_secondary_rws"], 
                  dict_node_index
              )

  G, positive_matrix = get_neighbor_matrix(G3, ngh_list, failed_nodes, walk_params_dict["num_neighbors"])

  node_list = list(G.nodes())

  negative_matrix = get_negative_examples(G, positive_matrix)

  pyg_graph = get_pyg_graph(G, train_set_fraction = 0.9)
  pyg_graph.node_list = node_list
  pyg_graph.positive_matrix = positive_matrix
  pyg_graph.negative_matrix = negative_matrix
   

  return( pyg_graph )



def scores(model, pyg_graph, labelled_data, seed_pages_used, criterion, edge_weight = False):
    '''
    Obtain validation loss, ranking and ranking score for an unsupervised GNN model.
    
    Inputs:
    - pyg_graph: torch geometric graph
    - neigh_matrix: matrix of positive samples
    - negative_array: matrix of negative samples
    - seed_pages_used: list of seed pages
    - edge_weight: boolean indicating whether esdges of the graph are weighted 
    
    Returns:
    - val_loss (float): negative sampling loss
    - scores_df_rankings: dataframe, pages are ordered by "max" score 
    - score: (float) standardised median rank difference between relevant and irrelevant pages.
    '''

    if edge_weight == True:
      h = model(pyg_graph.node_feature, pyg_graph.edge_index, pyg_graph.weight)
    else:
      h = model(pyg_graph.node_feature, pyg_graph.edge_index)

    out_numpy = h.detach().numpy()

    val_loss = criterion( h, pyg_graph.positive_matrix, pyg_graph.negative_matrix, pyg_graph.val_mask)

    scores_df = ranking_df_seeds(out_numpy, seed_pages_used, pyg_graph.node_list)



    scores_df_rankings = scores_df.sort_values(by = "max", ascending = False).reset_index(drop = False)
    scores_df_rankings.rename(columns = {"index": "page"}, inplace = True)

    score = calc_median_difference_n2v(scores_df_rankings, labelled_data, 
                            standardise = True, page_path = "page")
    
    return( val_loss, scores_df_rankings, score )



def ranking_df_seeds(out_numpy, seed_list, node_list):

  scores_df = pd.DataFrame(index =node_list)


  for node in seed_list:
    node_index = np.where(np.array(node_list) == node)

    node_embedding = out_numpy[node_index, :].squeeze()
    node_embedding_product = calc_cosine_similarity_matrix(out_numpy, node_embedding) # out_numpy @ node_embedding

    scores_df[node] = 0
    # print(scores_df.loc[:, node])
    print(node_embedding_product)
    scores_df.loc[:, node] = node_embedding_product

  scores_df["max"]= scores_df[seed_list].max(axis = 1)
  scores_df["median"] = scores_df[seed_list].median(axis = 1)
  scores_df["mean"] = scores_df[seed_list].mean(axis = 1)
  scores_df["min"]= scores_df[seed_list].min(axis = 1)

  return( scores_df )


def calc_cosine_similarity_matrix(X, y):
    '''
    X is a matatrix of embeddings, with nodes in rows (i.e number of rows = number of nodes, 
    number of columns = number of latent dimensions).
    '''
    cosine_similarity = np.dot(X, y)/(np.linalg.norm(X, axis = 1)* np.linalg.norm(y))
    
    return( cosine_similarity )

def get_model_path_name(hidden_channels, encoding_dim, dir_path):
    model_name = ""
    for i in hidden_channels: 
      model_name = model_name + str(i) + "_"
    model_name = model_name + "emb" + str(encoding_dim) + ".pt"
    path_save = dir_path + model_name

    return(model_name, path_save)


def load_pickle_file(file_name, dir_path="./data/outputs"):
    file_path = Path(dir_path + "/" + file_name)
    f = open(file_path, 'rb')
    file = pickle.load(f)
    f.close()

    return (file)