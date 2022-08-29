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


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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


def get_graph_from_transition_matrix(T, node_name_list = []):
    '''
    Create a graph object from a given transition matrix, with nodes named (in order) by the names
    in "node_name_list" (these would be webpage nameslinks in our case).
    
    Inputs:
    - T (np.array): transition matrix
    - node_name_list (list): list of graph node names (if left empty, nodes names are numbers)
    
    Returns:
    - networkx graph object
    '''
    G_from_T = nx.from_numpy_array(T, create_using = nx.DiGraph)
    
    if len(node_name_list) > 0: 
    
        node_relabel_dict = dict()

        for i in range( len(list(G_from_T.nodes)) ):
            node_relabel_dict[i] = node_name_list[i]

        G_from_T = nx.relabel_nodes(G_from_T, mapping = node_relabel_dict)
        
    else:
        pass
    
    return(G_from_T)



def add_ner_features(G, feature_df):
    '''
    Given graph object and dataframe of features, node features are added ("node_feature" element).
    Note that graph node names must correspond to the dataframe index to match nodes with their features.
    '''
    
    G_wfeatures = G.copy()
    
    nfeatures = feature_df.shape[1]
    nnodes = len(G_wfeatures.nodes())
    
    matrix_features = np.zeros( (nnodes, nfeatures) )

    # Add node features to the graph object
    omitted_nodes = []
    omitted_nodes_index = []

    for i, node in enumerate(G_wfeatures.nodes()):
        try:
            G_wfeatures.nodes[node]["node_feature"] = feature_df.loc[node, :]
            matrix_features[i, :] = feature_df.loc[node, :]
        except:
            G_wfeatures.nodes[node]["node_feature"] = np.zeros(nfeatures)
            matrix_features[i, :] = np.zeros(nfeatures)
            omitted_nodes.append(node)   
            omitted_nodes_index.append(i)
                 
    return( G_wfeatures, matrix_features, omitted_nodes, omitted_nodes_index )



def add_onehot_features(G, matrix_features, omitted_nodes_index = []):
    '''
    Add one hot encoded nodes to an existing graph and feature matrix
    '''
    
    G_wfeatures = G.copy()    
    
    nnodes = len(G_wfeatures.nodes())
    
    matrix_features_onehot = np.diag( np.ones( nnodes ) )
    
    if len(omitted_nodes_index) > 0:
        matrix_features_onehot = matrix_features_onehot[:,omitted_nodes_index]
    else:
        pass
    
    for i, node in enumerate(G_wfeatures.nodes()):
            G_wfeatures.nodes[node]["node_feature"] =  np.concatenate( (G_wfeatures.nodes[node]['node_feature'], matrix_features_onehot[i,:]) )
      
    matrix_features = np.concatenate( (matrix_features, matrix_features_onehot), axis = 1 )

    return( G_wfeatures, matrix_features )  


###########################################################


# Source: https://github.com/aditya-grover/node2vec (but modified)
class Graph():
    
    def __init__(self, nx_G, is_directed, p, q):
        self.G = nx_G
        self.is_directed = is_directed
        self.p = p
        self.q = q

    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]                            # current position (node)
            cur_nbrs = sorted(G.neighbors(cur))       # neighbs of current node
            if len(cur_nbrs) > 0:
                
                if len(walk) == 1:
                    # for the first step of the walk: no previous node
                    # hence we draw only from the connected edges
                    walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    # for further steps, precious node edges
                    # and hence w ecan use Node2Vec modified transition probabilities
                    prev = walk[-2]                   # previous position
                    next = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0],  
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk
    
    
    def simulate_walks(self, num_walks, walk_length, start_nodes = None):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        walks = []
        
        if start_nodes == None:
            start_nodes = list(G.nodes())
        else:
            pass
        
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(start_nodes)
            for node in start_nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))

        return walks
    
    def simulate_walks_from_seeds(self, num_walks, walk_length, seeds):
        
        walks_from_seeds = [self.simulate_walks(num_walks = num_walks, walk_length = walk_length, 
                                               start_nodes = (seed,) ) for seed in list(seeds)]
        
        
        
        pages_visited = {page for paths in walks_from_seeds for path in paths for page in path}

        results_rws = dict()
        results_rws["seeds"] = list(seeds)
        results_rws["pages_visited"] = pages_visited
        results_rws["paths_taken"] = walks_from_seeds

        
        return results_rws
    
    def convert_tfdf_to_n2v(self, results):
        
        walks_from_seeds = list(chain.from_iterable(results["paths_taken"]))
        
        return walks_from_seeds

    def get_alias_edge(self, src, dst):
        '''
        Get the alias edge setup lists for a given edge.
        
        src = source node
        dst = destination node
        '''
        G = self.G
        p = self.p
        q = self.q

        unnormalized_probs = []
        for dst_nbr in sorted(G.neighbors(dst)):
            if dst_nbr == src:
                # prob of returning to the source node
                unnormalized_probs.append(G[dst][dst_nbr]['edgeWeight']/p)
            elif G.has_edge(dst_nbr, src):
                # prob of going to node connected to both src and dst
                unnormalized_probs.append(G[dst][dst_nbr]['edgeWeight'])
            else:
                # prob of going further from src
                unnormalized_probs.append(G[dst][dst_nbr]['edgeWeight']/q)
        # normalise the new probabilities
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        G = self.G
        is_directed = self.is_directed

        # alias nodes are used to set up the alias edges (using ".get_alias_edge")
        alias_nodes = {}
        for node in G.nodes():
            # loop through all nodes
            unnormalized_probs = [G[node][nbr]['edgeWeight'] for nbr in sorted(G.neighbors(node))]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node] = alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}

        # now loop through edges in G
        if is_directed:
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
        else:
            # if undirected, set two values in each iteration (T is symmetric)
            for edge in G.edges():
                alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def alias_setup(probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    
    Mine: Efficient sampling from a multivariate discrete distribution, with a potentially
    unknown normalisign constant
    
    Source: https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    '''
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q



def alias_draw(J, q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    K = len(J)

    kk = int(np.floor(np.random.rand()*K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

    
    
def main(args):
    '''
    Pipeline for representational learning for all nodes in a graph.
    '''
    nx_G = read_graph()
    G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(args.num_walks, args.walk_length)
    learn_embeddings(walks)

    
    
def read_graph():
    '''
    Reads the input network in networkx.
    '''
    if args.weighted:
        G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not args.directed:
        G = G.to_undirected()

    return G



################ Node2Vec functions ####################

def learn_embeddings(walks):
    '''
    Learn embeddings by optimizing the Skipgram objective using SGD.
    '''
    walks = [map(str, walk) for walk in walks]
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.save_word2vec_format(args.output)

    return


def get_n2v_embeddings(model, pages):
    '''
    For a given N2V model, and a list of pages (page path), get the embedings
    for the supplied paths
    '''
    pages_ind = [l in pages for l in model.wv.index_to_key]
    pages_embeddings = model.wv.vectors[pages_ind,:]
    
    return(pages_embeddings, pages)


def get_n2v_scores(seed_pages_used, G, p, q, num_walks, walk_length, window, vector_size):
    '''
    For a given graph G and the parameters of node 2 vec algorithm, obtain rank pages based on the 
    median distance from seed_pages.
    
    If running for a multiple iterations, it is strongly recommended to get a coffee break - the preprocessing
    of transition probabilities results in a larg runtime.
    
    '''
    
    G_graph = Graph(G, is_directed = True, p = p, q = q)
    G_graph.preprocess_transition_probs()
    
    G_graph_results = G_graph.simulate_walks(num_walks = num_walks, walk_length = walk_length)
    model = Word2Vec(G_graph_results, window =window, vector_size = vector_size, compute_loss = True)
    
    rank, page_paths = calculate_n2v_distance(G, model, seed_pages_used, stats = np.max)
    print(model)
    page_rank_df = pd.DataFrame()
    page_rank_df["page"] = page_paths
    page_rank_df["score"] = rank
    page_rank_df.sort_values(by = "score", ascending = False, inplace = True)
    
    return( page_rank_df, model )


def get_min_mean_med_l2_distances(mm, seed_pages_used):
    
    page_rank_l2distance = pd.DataFrame()
    page_rank_l2distance["page"] = mm.wv.index_to_key

    for sp in seed_pages_used:
        vector_seed =  mm.wv.vectors[np.array(mm.wv.index_to_key) == sp]
        vector_seed_mat = np.repeat(vector_seed, mm.wv.vectors.shape[0], axis = 0)

        page_rank_l2distance["l2_" + sp] = np.linalg.norm(mm.wv.vectors- vector_seed_mat, axis = 1)
        
    page_rank_l2distance_min = page_rank_l2distance.copy()
    page_rank_l2distance_mean = page_rank_l2distance.copy()
    page_rank_l2distance_median = page_rank_l2distance.copy()
    
    page_rank_l2distance_min["score"] = page_rank_l2distance_min.iloc[:, -len(seed_pages_used):].min(axis = 1)
    page_rank_l2distance_mean["score"] = page_rank_l2distance_mean.iloc[:, -len(seed_pages_used):].mean(axis = 1)
    page_rank_l2distance_median["score"] = page_rank_l2distance_median.iloc[:, -len(seed_pages_used):].median(axis = 1)
    
    page_rank_l2distance_min.sort_values(by = "score", ascending = True, inplace = True)
    page_rank_l2distance_mean.sort_values(by = "score", ascending = True, inplace = True)
    page_rank_l2distance_median.sort_values(by = "score", ascending = True, inplace = True)
    
    return( page_rank_l2distance_min, 
            page_rank_l2distance_mean,
            page_rank_l2distance_median
          )


def calculate_l2_distance(embeddings, reference_vectors_embeddings):
    '''
    For given embeddings, and embeddings of reterence vectors, calulate mean and median distances
    from each embedding vector to reference vector embeddings.
    '''
    mean_distances = [np.linalg.norm(x - reference_vectors_embeddings, axis = 1).mean() for x in embeddings]
    median_distances = [np.median( np.linalg.norm(x - reference_vectors_embeddings, axis = 1) ) for x in embeddings]

    return(mean_distances, median_distances)

def calculate_n2v_distance(G, model, seeds, stats = np.max):
    '''
    For any given node in G, calculate similairty to each seed page and return the given summary 
    statistics of these scores.
    Returns scores for all nodes in G.
    '''

    rankings = list()
    page_paths = list()

    for node in G.nodes:

        rankings_node = list()

        for seed in seeds:
            rankings_node.append( model.wv.similarity(node, seed) )

        rankings.append( stats( rankings_node ) )
        page_paths.append( node )

    return( rankings, page_paths )


def calc_median_difference_n2v(df, labelled_data, standardise = True, page_path = "pagePath"):
    '''df needs to be a result of calling rw.page_freq_path_freq_ranking()
    
    df needs to be ranked from top page to the worst page (i.e. index represents ranking).'''
    df.reset_index(inplace = True, drop = True)
    
    df_labels = df.merge(labelled_data, left_on = page_path, right_on = "page path")
    df_labels.reset_index(inplace = True, drop = False)
    df_labels.rename(columns = {"index": "rank"}, inplace = True)

    med_ranking_label1 = df_labels[df_labels["label"] == 1]["rank"].median()
    med_ranking_label0 = df_labels[df_labels["label"] == 0]["rank"].median()
    
    if standardise == True:
        score = (med_ranking_label0 - med_ranking_label1) / ( df_labels[df_labels["label"] == 1]["rank"].std() +
                                                            df_labels[df_labels["label"] == 0]["rank"].std())
    else:
        score = med_ranking_label0 - med_ranking_label1
    
    return( score )

###################### GENERAL #########################


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

def set_seed(seed_number):
    '''Set random seeds for replicability'''
    random.seed(seed_number)
    np.random.seed(seed_number)

    return ()


############### DIM REDUCTION ###############


def fit_kmeans(data, n_components):
    scaler = StandardScaler()
    scaler.fit(data)
    data_standardised = scaler.transform(data)
    

    kmeans = KMeans(n_clusters=n_components)
    kmeans_fit = kmeans.fit(data_standardised)
    
    distances = kmeans_fit.inertia_
    labels = kmeans_fit.labels_
    centres = kmeans.cluster_centers_
    
    return(labels, distances, centres)
    
    

def squared_distances(model, n_components = range(1,10)):
    sum_of_squared_distances = []
    
    
    for k in n_components:
        _, _, distances, centres = reduce_dim_kmeans(model, k)
        sum_of_squared_distances.append(distances)
        
    return(sum_of_squared_distances)



def reduce_dim_kmeans(model, n_components):
    
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  
    
    kmean_labels, distances, centres = fit_kmeans(vectors, n_components)
    
    return(labels, kmean_labels, distances, centres)
    
    


def reduce_dimensions(model):
    num_dimensions = 2  # final num dimensions (2D, 3D, etc)

    # extract the words & their vectors, as numpy arrays
    vectors = np.asarray(model.wv.vectors)
    labels = np.asarray(model.wv.index_to_key)  # fixed-width numpy strings

    # reduce using t-SNE
    tsne = TSNE(n_components=num_dimensions, random_state=0)
    vectors = tsne.fit_transform(vectors)

    x_vals = [v[0] for v in vectors]
    if num_dimensions > 1:
        y_vals = [v[1] for v in vectors]
    else:
        pass
    return x_vals, y_vals, labels