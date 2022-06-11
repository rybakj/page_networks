import numpy as np
import pandas as pd
import networkx as nx
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from itertools import product
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

def group(original_list, n):
    '''Groups original_list into a list of lists, where each list contains n consecutive
    elements from the original_list'''
    return [original_list[x:x+n] for x in range(0,len(original_list),n)]


def evaluate(true_pages, predicted_pages, beta=2):
    '''
    true_pages is a list of all the pages known to belong to a WUJ
    predicted_pages ia a list of pages predicted to belong to a WUJ
    beta determines how much more important recall is than precision when computing fscore
    
    returns precision, recall and fscore
    '''
    
    true_pages = set(true_pages)
    predicted_pages = set(predicted_pages)
    
    # what proportion of true pages were correctly predicted?
    recall = len(true_pages.intersection(predicted_pages))/len(true_pages)
    
    # what proportion of predicted pages are true pages?
    precision = len(true_pages.intersection(predicted_pages))/len(predicted_pages)

    # compute f score, a harmonic mean of precision and recall
    fscore = ((1 + beta**2) * (precision * recall))/((precision * beta**2) + recall)
    
    return (precision, recall, fscore)

def getSlugs(G):
    '''
    Returns a list of slugs, given a networkx graph G.
    '''
    return [node[1]['properties']['name'] for node in G.nodes(data=True)]

def showGraph(G, k=None, iterations=50, node_size=100, figsize=None):
    '''
    Prints graph information and plots the graph, G.
    Takes figsize as input, a tuple, e.g. (10,10)
    '''
    print(nx.info(G))
    nx.spring_layout(G, k=k, iterations=iterations)
    plt.figure(figsize=(figsize))
    nx.draw(G, node_size=node_size)

def random_walk(A, G, steps, seed, p=False):
    '''
    A is an adjacency matrix, or a transition probability matrix. These should be CSR sparse matrices.
    Set p=True if using a transition probability matrix.
    G is a networkx graph.
    steps is the number of steps to take in the random walk.
    seed is a page slug for your starting node in the random walk. E.g. "/set-up-business" 
    
    returns a numpy array of node ids visited during the random walk.
    can return numpy array of nodes with their data if nodeData == True
    '''

    # set a seed node
    foundSeed = False
    for current_node_index, node in enumerate(G.nodes(data=True)):
        if node[1]["properties"]["name"] == seed:
            foundSeed = True
            break
    
    if not foundSeed:
        return []

    # list of nodes visited during the random walk
    visited = [current_node_index]
    
    transition_probs = None

    for _ in range(steps):

        # identify neighbours of current node
        neighbours = np.nonzero(A[current_node_index])[1]

        # if reached an absorbing state, i.e. no neighbours, then terminate the random walk
        if neighbours.size == 0:
            #print("Reached absorbing state after", step, "steps")
            visited = list(set(visited))
            return np.array(G.nodes())[visited]
        
        # if using transition probabilities, get them
        if p:
            transition_probs = A[current_node_index].toarray()[0, neighbours]
        
        # select the index of next node to transition to
        current_node_index = np.random.choice(neighbours, p=transition_probs)

        # maintain record of the path taken by the random walk
        visited.append(current_node_index)
    
    # return unique pages visited
    visited = list(set(visited))
        
    return np.array(G.nodes())[visited]

def M_walks_get_slugs(T,G,steps,repeats,seed_page,proba):
    '''Gets slugs from 'repeats' many random walks for a given seed page'''
    return [getSlugs(G.subgraph(random_walk(T,G,steps,seed_page,proba))) for _ in range(repeats)]

def check_seed_pages(seeds, G):
    G_nodes = set([node[1]['properties']['name'] for node in G.nodes(data=True)])
    not_found = [seed for seed in seeds if seed not in G_nodes]
    if len(not_found) > 0:
        print(not_found, 'could not be found in the graph')
        return not_found
    else:
        return []
    
def repeat_random_walks(steps, repeats, T, G, seed_pages, proba, combine, level=0, verbose=1, n_jobs=1):
    '''
    Performs 'repeats' many random walks per seed page in seed_pages, each with 'steps' many steps. seed_pages is a list
    of page slugs. e.g. 
    ['/government/collections/ip-enforcement-reports',
    '/government/publications/annual-ip-crime-and-enforcement-report-2020-to-2021',
    '/search-registered-design']
    
    Each random walk will traverse a network of gov.uk pages, each one recording the set of pages
    that were visited.
    
    repeats*len(seed_pages) many sets of pages will be generated.
    
    combine takes value 'intersection' or 'union',
    depending on whether to compute the union or intersection of these sets.
    If combine is set to 'no', then a list of len(seed_pages) lists will be
    returned. Each list will contain the paths of the 'repeats' many random walks performed per seed node.

    if combine = 'union' or 'intersection:
    level = 0 unions/intersects the pages visited by all 'repeats' many random walks
    at the level of seed nodes, giving you one set of pages per seed node.

    E.g. Suppose combine = 'union', level = 0, repeats = 2, and we have two seed nodes A and B.
    From A, two random walks are performed, following the paths
    [A,C,X,G,X,A] and [A,D,X,A]
    These combine to become {A,C,D,G,X}

    From B, two random walks are performed, following the paths
    [B,O,P,L,D] and [B,O,M,N]
    These combine to become {B,D,L,M,N,O,P}

    Hence, we get a list of these two sets [{A,C,D,G,X}, {B,D,L,M,N,O,P}]

    level = 1 unions/intersects the pages visited by all repeats*len(seed_pages) random walks,
    giving you one set of pages.

    E.g. Suppose combine = 'union', level = 1, repeats = 2, and we have two seed nodes A and B.
    From A, two random walks are performed, following the paths
    [A,C,X,G,X,A] and [A,D,X,A]
    From B, two random walks are performed, following the paths
    [B,O,P,L,D] and [B,O,M,N]
    These four paths merge into a single set:
    {A,B,C,D,L,M,N,O,P,X}
    
    T is an adjaceny matrix or a transition probability matrix. They are CSR sparse matrices.
    If using a probability transition matrix, set proba=True.

    verbose >= 1 if you want progress bars. verbose <= 0 if you don't want progress bars.
    
    n_jobs is the number of CPUs to use.
    For large experiments, I recommend n_jobs = -2, to use all but 1 of your CPUs, leaving
    1 CPU available for other tasks.
    For small experiments, I recommend n_jobs = 1. The overhead of n_jobs > 1 is only
    worth it for large experiments, e.g. when repeats > 100.
    '''

    # find seed pages not found in the graph
    not_found = set(check_seed_pages(seed_pages, G))

    # remove seed pages not found in the graph
    seed_pages = [page for page in seed_pages if page not in not_found]

    # for each seed node, compute paths taken
    if verbose >= 1:
        paths_taken = Parallel(n_jobs=n_jobs)(delayed(M_walks_get_slugs)(T,G,steps,repeats,seed_page,proba) for seed_page in tqdm(seed_pages))
    else:
        paths_taken = Parallel(n_jobs=n_jobs)(delayed(M_walks_get_slugs)(T,G,steps,repeats,seed_page,proba) for seed_page in seed_pages)
    
    if combine == 'union':
        if level == 0:
            pages_visited = [set([page for path in paths for page in path]) for paths in paths_taken]
        elif level == 1:
            pages_visited = {page for paths in paths_taken for path in paths for page in path}

    elif combine == 'intersection':
        if level == 0:
            pages_visited = [set.intersection(*map(set,paths)) for paths in paths_taken]
        elif level == 1:
            pages_visited = set.intersection(*[set([page for path in paths for page in path]) for paths in paths_taken])

    elif combine == 'no':
        pages_visited = paths_taken

    else:
        print(combine, 'is an invalid path combination method')
        return

    if not pages_visited:
        print("No pages found")
        return

    return {'seeds': seed_pages, 'pages_visited': pages_visited, 'paths_taken': paths_taken}

def M_N_Experiment(steps, repeats, T, G, target_pages, seed_pages, proba, n_jobs):
    '''
    For a given transition matrix T, graph G, set of WUJ target_pages and seed_pages within a WUJ,
    this function tries every combination of steps and repeats. E.g.

    Number of steps to take in random walk
    steps = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600]

    Number of times to initialise random walk from a given seed node
    repeats = [10,20,30,40,50,60,70,80,90,100,200,300,400,500,600]

    Set proba=True if T contains probabilities, and proba=False if T is an adjacency matrix.

    n_jobs = number of workers to use during execution, for parallelisation.
    '''
    # all combinations of N and M
    NMs = list(product(steps,repeats))

    results = Parallel(n_jobs=n_jobs)(delayed(repeat_random_walks)(step, repeat, T, G, seed_pages, proba, 'union', 1, 0, 1) for step, repeat in tqdm(NMs))

    scores = []
    for i, result in enumerate(results):
        path = result['pages_visited']
        p, r, f = evaluate(target_pages, path)
        n, m = NMs[i]
        scores.append([p,r,f,n,m,len(path)])

    return scores


def page_freq_path_freq_ranking(results):
    '''
    Args:
        results (dict): the dictionary returned after running repeat_random_walks
                        E.g. results = repeat_random_walks(steps=100, repeats=100, T=A, G=G, seed_pages=seeds, proba=False, combine='union', level=1, n_jobs=1)

    Return:
        page_scores (Pandas dataframe): a dataframe of page paths, where the page paths are ranked by the page frequency-path frequency metric.

    '''
    # create a list of paths [path1, path2, path3, etc], where each path is a list of pages
    paths = [path for paths in results['paths_taken'] for path in paths]

    # map each page to a number {pageA: 0, pageB: 1, pageC: 2, etc...}
    # this corresponds to the column index in the array called 'tf' below
    pages_visited = {page: i for i, page in zip(range(len(results['pages_visited'])), results['pages_visited'])}

    # initialise an array for storing the scores
    # each page gets a page-freq path-freq score per path
    # each row corresponds to a path and each column corresponds to a page
    tf = np.zeros((len(paths), len(pages_visited)))

    # tf = 
    # +-------+-------+-------+-------+
    # |       | pageA | pageB | pageC |
    # +-------+-------+-------+-------+
    # | path1 |       |       |       |
    # +-------+-------+-------+-------+
    # | path2 |       |       |       |
    # +-------+-------+-------+-------+
    # | path3 |       |       |       |
    # +-------+-------+-------+-------+


    # compute the page frequency for all pages on each path
    for row, path in enumerate(paths):
        for page in path:
            tf[row, pages_visited[page]] += 1

    # compute the number of paths each page occurs on
    df = np.where(tf > 0, 1, 0).sum(axis=0)

    # compute a page frequency-path frequency score for each page on each path
    tfdf = tf @ np.diag(df)

    # aggregate the page frequency-path frequency scores into a single number per page
    tfdf_sums = tfdf.sum(axis=0)
    tfdf_max = tfdf.max(axis=0)
    tfdf_mean = tfdf.mean(axis=0)

    # construct a data frame of pages and their scores
    page_scores = [(page, tfdf_sums[row], tfdf_max[row], tfdf_mean[row]) for page, row in pages_visited.items()]
    page_scores = pd.DataFrame.from_records(page_scores, columns=['pagePath','tfdf_saliency','tfdf_max','tfdf_mean'])

    # rank by tfdf_max
    page_scores.sort_values(by='tfdf_max', inplace=True, ascending=False)

    return page_scores


def get_transition_matrix(G):
    '''
    Computes a transition probability matrix for a graph, using normalised edge weights.

    Args:
        G: a weighted networkx graph
    
    Return:
        csr_matrix(T_probs): a transition probability matrix as a a csr matrix
    '''

    # Create array with edge weight
    T = nx.adjacency_matrix(G, weight="edgeWeight").todense()
    T_array = np.array(T)

    # Transform edge weight into probabilities

    # Normalisation
    sum_of_rows = T_array.sum(axis=1)
    T_probs = T_array / sum_of_rows[:, np.newaxis]

    # Rows with only 0s = nan. Replace nan values with 1/Tarray.shape[0]
    np.nan_to_num(T_probs, nan=1 / T_array.shape[0], copy=False)

    # Convert into a transition matrix (for random walks function)
    return csr_matrix(T_probs)

def reformat_graph(G):
    '''
    Reformat the graph to make it compliant with existing random walk functions
    i.e. add the path to a name property and set the index to be a number.

    Args:
        G: networkx graph
    
    Return:
        G: networkx graph
    '''

    for index,data in G.nodes(data=True):
        data['properties'] = dict()
        data['properties']['name'] = index

    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default', label_attribute=None)

    return G

def add_additional_information(page_scores, G):
    '''
    Add additional information to the random walk output: 
    - document type
    - document super type
    - number of sessions that visit this page
    - number of sessions where this page is an entrance hit
    - number of sessions where this page is an exit hit
    - number of sessions where this page is both an entrance and exit hit
    - how frequent the page occurs in the whole user journey

    Args:
        page_scores: pandas dataframe returned by `page_freq_path_freq_ranking()`
        G: networkx graph

    Return:
        df_merged: pandas dataframe with additional information
    '''

    # Document supertypes
    news_and_comms_doctypes = {
        "medical_safety_alert",
        "drug_safety_update",
        "news_article",
        "news_story",
        "press_release",
        "world_location_news_article",
        "world_news_story",
        "fatality_notice",
        "fatality_notice",
        "tax_tribunal_decision",
        "utaac_decision",
        "asylum_support_decision",
        "employment_appeal_tribunal_decision",
        "employment_tribunal_decision",
        "employment_tribunal_decision",
        "service_standard_report",
        "cma_case",
        "decision",
        "oral_statement",
        "written_statement",
        "authored_article",
        "correspondence",
        "speech",
        "government_response",
        "case_study",
    }

    service_doctypes = {
        "completed_transaction",
        "local_transaction",
        "form",
        "calculator",
        "smart_answer",
        "simple_smart_answer",
        "place",
        "licence",
        "step_by_step_nav",
        "transaction",
        "answer",
        "guide",
    }

    guidance_and_reg_doctypes = {
        "regulation",
        "detailed_guide",
        "manual",
        "manual_section",
        "guidance",
        "map",
        "calendar",
        "statutory_guidance",
        "notice",
        "international_treaty",
        "travel_advice",
        "promotional",
        "international_development_fund",
        "countryside_stewardship_grant",
        "esi_fund",
        "business_finance_support_scheme",
        "statutory_instrument",
        "hmrc_manual",
        "standard",
    }

    policy_and_engage_doctypes = {
        "impact_assessment",
        "policy_paper",
        "open_consultation",
        "policy_paper",
        "closed_consultation",
        "consultation_outcome",
        "policy_and_engagement",
    }

    research_and_stats_doctypes = {
        "dfid_research_output",
        "independent_report",
        "research",
        "statistics",
        "national_statistics",
        "statistics_announcement",
        "national_statistics_announcement",
        "official_statistics_announcement",
        "statistical_data_set",
        "official_statistics",
    }

    transparency_doctypes = {
        "transparency",
        "corporate_report",
        "foi_release",
        "aaib_report",
        "raib_report",
        "maib_report",
    }

    # Create a df with `pagePath`: `documentType`, `sessionHitsAll`, `entranceHit`, `exitHit`, `entranceAndExitHit`
    df_dict = {
        info["properties"]["name"]: [
            info["documentType"],
            info["sessionHitsAll"],
            info["entranceHit"],
            info["exitHit"],
            info["entranceAndExitHit"],
            info["sessionHits"],
        ]
        for node, info in G.nodes(data=True)
    }
    df_dict = {
        k: v for (k, v) in df_dict.items() if k in page_scores["pagePath"].tolist()
    }
    df_info = (
        pd.DataFrame.from_dict(
            df_dict,
            orient="index",
            columns=[
                "documentType",
                "sessionHitsAll",
                "entranceHit",
                "exitHit",
                "entranceAndExitHit",
                "sessionHits",
            ],
        )
        .rename_axis("pagePath")
        .reset_index()
    )

    # Create a df with document supertypes
    document_type_dict = dict.fromkeys(list(set(df_info["documentType"])))

    for docType, docSupertype in document_type_dict.items():
        if docType in news_and_comms_doctypes:
            document_type_dict[docType] = "news and communication"

        elif docType in service_doctypes:
            document_type_dict[docType] = "services"

        elif docType in guidance_and_reg_doctypes:
            document_type_dict[docType] = "guidance and regulation"

        elif docType in policy_and_engage_doctypes:
            document_type_dict[docType] = "policy and engagement"

        elif docType in research_and_stats_doctypes:
            document_type_dict[docType] = "research and statistics"

        elif docType in transparency_doctypes:
            document_type_dict[docType] = "transparency"

        else:
            document_type_dict[docType] = "other"

    df_docSuper = pd.DataFrame(
        document_type_dict.items(), columns=["documentType", "documentSupertype"]
    )

    # Merge dfs
    df_merged = pd.merge(page_scores, df_info, on="pagePath")
    df_merged = pd.merge(df_merged, df_docSuper, how="left")

    # Reoder and rename df columns
    df_merged = df_merged[
        [
            "pagePath",
            "documentType",
            "documentSupertype",
            "sessionHitsAll",
            "entranceHit",
            "exitHit",
            "entranceAndExitHit",
            "sessionHits",
            "tfdf_max",
        ]
    ]
    df_merged = df_merged.rename(
        columns={
            "pagePath": "page path",
            "documentType": "document type",
            "documentSupertype": "document supertype",
            "sessionHitsAll": "number of sessions that visit this page",
            "entranceHit": "number of sessions where this page is an entrance hit",
            "exitHit": "number of sessions where this page is an exit hit",
            "entranceAndExitHit": "number of sessions where this page is both an entrance and exit hit",
            "sessionHits": "all sessions that visit this page, regardless of the session visiting a seed page",
            "tfdf_max": "how frequent the page occurs in the whole user journey",
        }
    )

    return df_merged