Graph files can be found here: https://drive.google.com/drive/folders/1bLMolKJoRzJKzIEQMTjcWfE0qF_VsoeN?usp=sharing

# Repo structure:
- `pickle_graph_objects.ipynb`: loads and processes initial graph objects (these are then used as starting points for all models)
- `processing_labelled_sets.ipynb`: simple manipulation of labelled sets
- `method_comparison.xlsb`: Contains top 50 pages for all approaches considered here and, based on these, the evaluation figure displayed below (see "Evaluation" section)
- `/NER`: contains notebooks and scripts for named-entity-recognition
- `/context_approaches`: notebooks for context-based approaches, i.e. original approach and Node2Vec
- `gnns`: notebooks for semi-supervised and unsupervised gnn
- `src`: contains functions used by notebooks throughout the repor (especially context-based and unsupervised GNN notebooks).


# Whole User Journey (WUJ) analysis for gov.uk

The aim of this project is to identify all of the pages relevant to a whole user journey in gov.uk. We focus on "economic recovery" use case throughout our analysis.

## Structure

In Section 1 we describe how webpages are used to construct probabilistic graph. Section 2 categorises different approaches to webpage ranking. Section 3 presents the existing approach to identifying relevant webpages, which serves as our benchmark, while Section 4 outlines methods we considered to improve upon this benchmark. Section 5 describes how named entity recognition data were obtained and Section 6 contains overview of main results.

## 1. Inputs

Starting with an expert-compiled list of webpages deemed relevant for a user journey, a probabilistic graph of gov.uk webpages is constructed. We use the existing approach to constructing such graph, which is briefly described below. For our analysis, we load a probabilistic graph already created by the DataLabs team.

Steps:
1. Start with "seed" pages identified as relevant to a given user journey.
2. Obtain pages hyperlinked from the seed pages.
3. Use BigQuery to extract user movement to and from the pages from steps (1) and (2).

Specifically, in step (3) we pick all sessions where user selects at least one webpage from steps (1) and (2) and retrieve all webpages visited during these sessions. 

Output: Probabilistic graph of pages
- All webpages that appear in the BigQuery are used as graph nodes.
- Edge weights are the probabilities of moving from one page to another (calculated using the number of such moves in step (3)).


## 2. Solution space

We can categorise solutions based on what information is used to rank webpages. In our context, this can be phrased as a question of "what makes webpages relevant to a user journey". As we chracterise the user journey by seed pages selected by an expert, this problem can be phrased as "what makes two webpages similar" (in a sense relevant to a user journey). Based on this, two main approaches (i.e. two "similarity hypotheses") can be characterised:

What makes pages / nodes similar?
- Context: pages with a similar context tend to be similar     
- Content: pages with similar content tend to be similar

### 2.1 Context-based approaches

*Hypothesis: nodes in a similar context tend to be similar*

Context can be defined in various ways. In the context considered here, this can mean, for example, that if users who visit webpage A often proceed to visiting webpage B, the webpages A and B have similar relevance to a user journey. Alternatively, webpages which appear in many user paths extracted by BigQuery can be seen as (equally) relevant to a given user path.

In the language of graphs these two cases can be can be respectively phrased as:
- Homophily: nodes that are highly connected or belong to similar clusters have similar relevance.
- Structural equivalence: nodes with similar structural roles have similar relevance.

In the figure below, the yellow node represents a seed node. Homophily hypothesis states that yellow and green nodes have similar relevance, while structural equivalence states that yellow and red nodes have similar relevance.

![image](https://user-images.githubusercontent.com/71390120/184164444-81a31ac2-30e0-4b17-87e8-a49fa8aae548.png)

### Extracting context

One possibility of extracting context of a graph node (i.e. webpage) is to use random walks. Specifically, starting in a given node, say A, a subsequent node is selected from the neighbors of node A randomly. Different types of random walks exist. They differ in whether homophily or structural equivalence are assumed, and based on this the transition probabilities (i.e. probability of selecting a given neighbor of node A) are modified, in order to bias the random walks to explore the relevant part of the graph (neighborhoods under homophily, far-away nodes under structural equivalence).

### 2.2 Content-based approaches

*Hypothesis: nodes with a similar content tend to be similar*

The shortcoming of context-based approaches is that they ignore the content of webpages. Since webpages with similar content are likely to cover related topics, and thus have similar relevance to a user journey, incorporating content could be beneficial for webpage ranking. 

This approach consists of two steps:
1. Content extraction: in our case, we will extract named entities from pages.
2. Content comparison: represent the extracted content as vectors and combine them with the graph representation.

## 3. Current approach

Starting from the probabilistic graph described in Section 1, the ranking is created as follows:
 1. Run undirected random walks from seed nodes (100 walks of length 100 from each node) and record pages visited along these walks.
 2. Use "page-frequency-path-frequency" metric to rank the relevance of the pages in the journey. For a given webpage this metric combines the number of walks it occurs on and the (maximum) number of times it occurs within a single random walk. Specifically, for a given webpage, say B, PFPF score is given by

$$ \text{PFPF(B)} = \text{number of random walks page B occurs on} \times \text{max number of occarences of page B within a single random walk.} $$

The current approach is thus context-based. As this approach lets random walks wander far from the starting node, it is likely to be biased towards structural equivalence assumption (although a more detailed analysis would be needed to confirm this).

The whole existing approach, from BigQuery to final rank creation is illustrated on the figure below. In red are the part we seek to modify.

<img src="https://user-images.githubusercontent.com/71390120/184177371-333ede2b-5d04-4292-a6fb-96f7294dfd8e.png" width=50% height=50%>

## 4. Our approach

We first consider an alternative context-only approach, based on second-order random walks (Section 4.1), introduced by Grover et al. (2016). Subsequently, we combine context and content-based approaches using graph neural networks (Section 4.2). 

A common feature of our methods is that we seek to encode graph nodes (webpages) as vectors. That is, starting from a graph, for a given node (say node u in the picture below), we seek a mapping of that node into a vector space.

![image](https://user-images.githubusercontent.com/71390120/184178702-f6a3e3b3-216a-4b58-a9c7-46b24b3c742e.png)

*Source: Hamilton, 2020*

This is hardly a surprising feature. Indeed, even the current method embeds nodes as vectors (specifically, each node is encoded in a vector of length 100, with elements given by the counts of the corresponding website in each random walk). What is different in our approach is that this vector is obtained as a solution to an **optimisation problem**, as opposed to a heuristic rule.

### 4.1 Context-based approaches

We modify the original procedure in three ways.
1. Introduce second-order random walks.
2. Vector embeddings are a solution to an optimisation problem.
3. Cosine simularity is used as a similarity metric.

Overall this corresponds to Node2vec (Grover et al, 2016). 

The second-order random walks modify the way in which node context is sampled. By varying the choice of hyperparameters, second-order random walks can focus on exploring starting node's neighbours (graph "breadth") or wander far from the starting node (exploring the network "depth"), and to interpolate between these two approaches. 

<img src=https://user-images.githubusercontent.com/71390120/184365359-5858f189-d939-458c-a1c3-b79f88e37bd3.png width=40% height=40%>

In the first step, we only modify random walks and leave the original PFPF ranking method intact. In the second step we run the full Node2vec procedure, that is the second-order random walks together with new node embeddings and a new ranking metric.

In order to compare the three methods (original, 2nd order RWs + PFPF metric, and Node2vec) we run the original ranking procedure (with a random seed) and manually label top 100 pages as either relevant or irrelevant to a user journey.

We evaluate the three methods using the following score (*higher score = better*)

$$ \frac{ \text{median(irrelevant)} - \text{median(relevant)} }{ \sigma( \text{irrelevant} ) + \sigma( \text{relevant} ) } $$

where "relevant" is a ranking (top = 1, bottom = 100) of pages labelled as relevant to a user journey, and similarly for "irrelevant" , and $ \sigma $ is a standard deviation.

The original method achieves score of around 0, while the 2nd order RWs + PFPF metric achieves a score of 0.22 and Node2vec achieves the score of 0.24 (the latter two averaged over multiple initialisation and hyperparameter choices). As all three methods are random in nature, we average scores over 10 runs (for each hyperparameter choice).

Crucially, the higher scores rely on breadth-first search, that is on random walks exploring starting node's neighborhood first (the green arrows in the figure above).
We will make use of this observation when formulating unsupervised approaches combining node context and content.

#### Shortcomings of context-based approaches

It is widely recognised that methods such as Node2vec suffer from a number of drawbacks (see e.g. Hamilton, 2020). In our setting the most relevant one are:
1. Methods are not applicable to unseen nodes: If the BigQuery is ran over a different time period (e.g. more recent), new webpages (i.e. new nodes) are likely to be present. The random-walk based approaches discussed here cannot calculate ranking for such new nodes and the whole procedure needs to be re-fitted.
2. These methods don't consider node features, e.g. the content of webpages. The ranking is thus based purely on the node context, i.e. on the user movement between webpages.

### 4.2 Context and content-based approaches

We now seek to combine both webpage context and content to create a ranking. For content, we use entities appearing on each webpage as node (webpage) features (see next section for more details). Given the node features, we use graph neural networks (GNNs) to combine node context and node features (content) in a single model.

GNNs are based on an idea of message passing, where node features are updated to incorporate features of the neighboring nodes. Consider a simple directed graph in the figure below.

In the first stage of message passing, each node aggregates the features of the neighboring nodes. For example, the blue node aggregates node features of its neighbors (in green). Likewise, each of green nodes aggregatse features of its neighbors (in yellow). As a result, we obtain a new graph (on the right), with the same structure but different node features. The blue node will now contain a combination of its own features and features of green nodes (hence the node is two-coloured now), etc.
This corresponds to a one-layer GNN (a two-layer GNN would repeat the same step of feature aggregation on the graph obtained from the first layer). 

<img src=https://user-images.githubusercontent.com/71390120/184371924-ae5338e4-f90e-49ec-99be-6ee32a49032a.png width=80% height=80%>

There are various ways we can combine the features of node's neighbors with its own features, and this results in different GNN architectures. Once the architecture is selected, the parameters are optimised in the usual way, as a function minimisation. The objective function that is minimised is another crucial difference between different GNN methods.

Throughout our analysis we keep the architecture fixed and use convolutional GNNs (Kipf and Welling, 2016). We consider two approaches:
1. Semi-supervised: we manually label part of the nodes and train GNNs to solve a classification problem using this subset of nodes.
2. Unsupervised: Use encoder-decoder models to rank nodes.



#### Node features: Named-entity recognition 
To enrich the nodes (webpages) from gov.uk with features prior to implementing GNNs, inference was performed from a previously created Named-Entity Recognition (NER) model which utilises the DistillBERT architecture (Sanh et al. 2019). This model can be used to highlight named-entities in a number of categories (e.g., organisation, people) which can be visualisd as follows: 

<img width="1741" alt="Screenshot 2022-06-16 at 19 19 49" src="https://user-images.githubusercontent.com/104083260/174139092-45e6010a-2462-4e46-b227-d55c6418605a.png">

The NER script in this repo can be used to output data into a .csv file in the following format:
<img width="297" alt="Screenshot 2022-06-16 at 19 21 37" src="https://user-images.githubusercontent.com/104083260/174139369-87196229-ec26-4185-a21c-44ac116470b9.png">


#### Semi-supervised approach

Semi-supervised approaches aim to label all graph nodes, given a small fraction of labelled nodes, which are used to evaluate a loss function.

We use a small set of nodes (500 nodes ~ 5%), which are the top-ranking pages of the original algorithm, to train a Graph Convolutional Network (GCN) model. The model achieves accuracy of 76% on a hold-out set, which is similar to performance achieved on widely-used dataset, for example the Cora citation dataset (accuracy of 81%).

We have also experimented with other models such as the Graph Attention Network (GAT), but these achieved lower prediction accuracy.


#### Unsupervised approach

An unsupervised GNN can be thought of as an encoder-decoder model. Encoder embeds graph nodes into a vector space (as described above). Then, a decoder uses vector embeddings to reconstruct a certain property of the nodes. That is, starting from vector embeddings decoder aims to recontsruct a certain statistic of the nodes. The figure below illustrates this.

<img src=https://user-images.githubusercontent.com/71390120/184417829-5ab58787-862e-454d-9959-9447bbb86433.png width=80% height=80%>

*Source: Hamilton, 2020*

Unsupervised GNN models differ in the construction of the encoder and the statistic that the model is aiming to reconstruct. The statistic is chosen so that the model suits the application at hand.

Our choice of statistics has been motivated by the encouraging results of Node2Vec described earlier. Specifically, we aim to maximise the probability of observing node's neighbors given node's embedding. As this objective function is infeasible to calculate, we use second order random walks for negative sampling, as in Node2Vec. As an encoder, we use a two-layer convolutional GNN. Also in a direct analogy with Node2vec, we use cosine distance as a ranking metric. A more detailed description of this approach is provided in `semi-supervised.ipynb` notebook.

# Evaluation

In order to compare different methods of ranking webpages, we proceed as follows:
1. For each method obtain top 50 webpages (i.e. the webpages ranked as most relevant to the whole user journey)
2. Label webpages from step 1 as relevant (value 1) or irrelevant (value 0)
3. Calculate rolling proporton of relevant pages.

As slready mentioned we focus on "economi recovery" whole user journey. However, in many cases we found it hard to decide whether a given webpage is relevant or irrelevant to the user journey. For this reason, we use two different sets of labels (one set labelled by Douglas, one by Jakub) and report the average of these two labelled sets. Overall, we label 300 webpages, of which 53% are labelled as relevant in one set of labels, while 60% are deemed relevant in another set of labels.

The figure below shows the % of relevant pages (y-axis) within a given number of top pages (x-axis), for each method.
<img src=https://user-images.githubusercontent.com/71390120/187199771-28b2fd66-289d-45aa-844f-a48142005922.png width=70% height=70%>

As we can see, Node2Vec performs significantly better than other methods. This seems to be due to its bias to explore neighboring nodes of a starting node (i.e. seed node). For example, when seed nodes are: `/find-a-job`, `/universal-credit`, and `/government/collections/financial-support-for-businesses-during-coronavirus-covid-19`, the top 20 pages are as follows:
    
![image](https://user-images.githubusercontent.com/71390120/187200192-b2aec9e8-e699-4e27-9ad5-dc657e2d4004.png)


A potential disadvantage is that webpages relevant to a user journey, but not falling into the three categories represented by the seed pages, are not uncovered.

# References

Grover, A. and Leskovec, J., 2016, August. node2vec: Scalable feature learning for networks. In Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining (pp. 855-864).

Hamilton, W.L., 2020. Graph representation learning. Synthesis Lectures on Artifical Intelligence and Machine Learning, 14(3), pp.1-159.

Kipf, T. N. and Welling, M., 2016 Semi-supervised classification with graph convolutional networks.
In ICLR, 2016.

Sanh, V., Debut, L., Chaumond, J. and Wolf, T., 2019. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. arXiv preprint arXiv:1910.01108.
