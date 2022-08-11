Graph files can be found here: https://drive.google.com/drive/folders/1bLMolKJoRzJKzIEQMTjcWfE0qF_VsoeN?usp=sharing

Note:
- `er` stands for economic recovery. 
- `Adjacency_er` and `Transition_er` refer to the matrices associated with a directed probabilistic graph on ER webpages.
- `Graph_er` is a graph object (unweighted edges, no features). 
- `Graph_er_weighted`: same graph object, with edge weights corresponding to node transition probabilities.
- `Graph_er_weighted_wfeatures`: node features added on top of the graph object.




# Whole User Journey (WUJ) analysis for gov.uk

The aim of this project is to identify all of the pages relevant to a whole user journey in gov.uk. We focus on "economic recovery" use case throughout our analysis.

## Inputs

In order to answer this question, we turn the webpages of gov.uk into a graph as follows.

Steps:
1. Start with "seed" pages identified as relevatn to a given user journey.
2. Obtain pages hyperlinked from the seed pages.
3. Use BigQuery to extract user movement to and from the pages from steps (1) and (2).

Specifically in step (3) we pick all sessions where user selects at least one webpage from steps (1) and (2) and retrieve all webpages visited during these sessions. 

Output: Probabilistic graph of pages
- All webpages that appear in the BigQuery are used as graph nodes.
- Edge weights are the probabilities of moving from one page to another (calculated using the number of such moves in step (3)).


## Solution space

The solutions to this problem can be cateogirsed based on what information is used to rank pages. In other words, this can be phrased as a question of "what makes webpages relevant to a user journey". As we chracterise the user journey by seed pages selected by an expert, this problem can be phrased as "what makes two webpages similar" (in a sense relevant to a user journey). Based on this, two main approaches (i.e. two "similarity hypotheses") can be characterised:

What makes pages / nodes similar?
- Context: pages with a similar context tend to be similar     
- Content: pages with similar content tend to be similar

### Context-based approach

***Hypothesis: nodes in a similar context tend to be similar***

Context can be defined in various ways. In the context considered here, this can mean, for example, that if users who visit webpage A often proceed to visiting webpage B, the webpages A and B have similar relevance to a user journey. Alternatively, webpages which appear in many user paths extracted by BigQuery can be seen as (equally) relevant to a given user path.

In the language of graphs these two cases can be can be respectively phrased as:
- Homophily: nodes that are highly connected or belong to similar clusters have similar relevance.
- Structural equivalence: nodes with similar structural roles have similar relevance.

In the figure below, the yellow noode represents a seed node. Homophily hypothesis states that yellow and green nodes have similar relevance, while structural equivalence states that yellow and red nodes have similar relevance.

![image](https://user-images.githubusercontent.com/71390120/184164444-81a31ac2-30e0-4b17-87e8-a49fa8aae548.png)


## Current approach

1. Use seed pages (and pages linked from the seed pages) to construct a probabilistic graph of user movement using BigQuery.
  The edge weights of this graph correspond to probabilitiy of a user moving from one page to another (this is obtained by extracting the number of such moves using BigQuery).
 2. Use undirected random walks from seed nodes (100 walk from each node) and record pages visited along these walks.
 3. Use "page-frequency-path-frequency" metric to rank the relevance of the pages in the journey. For a given webpage this metric combines the number of walks it occurs on and the (maximum) number of times it occurs within a single random walk.

## Our approach

### As a modification of current approach




We tried various approaches to this problem. They can all be seen as a successive modification of the curent approach.

1. Replace random walks in the original approach by second-order random walks
2. 

The current approach utilises random walks which begin from a small set of seed nodes in an undirected graph. The "page-frequency-path-frequency" metric is utilised to rank the relevance of the pages in the journey. 

We proposed a number of changes to this approach:
- Use a directed graph, rather than an undirected graph
- Utilise biased random walks (homophily vs equivariance)
- Utilise Word2Vec embedding
- Extract named-entities from gov.uk pages to enrich the nodes with metadata
- Use graph neural networks to improve the edge-level predictions

## Contents
1. Random walk approaches
2. Named-entity recognition
3. Graph neural networks

## Random walk approaches
### Performance metric

To compare different ranking algorithms we use the following performance measure.

For a single run of the original ranking algorithm, we produce a list of top 100 pages. These are manually labelled (1 = relevant, 0 = irrelevant) and used for algorithm evaluation ("evaluation set") using the following metric:

$$ \text{score} = \frac{ \text{median}(relevant) - \text{median}(irrelevant)}{ \sigma(relevant) + \sigma(irrelevant) }, $$

where $ relevant $ and $ irrelevant $ are rankings of evaluation set pages labelled 1 and 0 respectively, and $ \sigma $ is a standard deviation.

All of the apporaches discussed here involve some randomness, predominantly coming from random walks. For this reason, scores given below are averaged over multiple runs (usually 10 or 20) of the same method for the same combination of hyperparameters. 

### Evaluation of proposed changes

To choose which of the proposed steps has positive impact on rankings we ran the following ranking algorithms.

1. **Biased random walks**: Use biased RWs (from the same seeds, using the same walk length, and the same ranking metric (pfpf)). This measures proposed changes 1 & 2.
2. **Biased random walks & vector embeddings**: Use N2V embedding to vector space. Metric is a simple L2 norm distance from the same seeds as original ranking procedure.
3. **Biased random walks & vector embeddings & W2V scores** (this corresponds to Node2Vec approach).

The original apporach (undirected graph, unbiased RWs, pfpf score) gives a score of close to zero (with a meaon of **-0.016** and standard deviation 0.033). This constitutes our benchmark.

The scores for different parameter combinations for the three methods can be found below. While certain parameter combinations are far from ideal (or sensible), the overall improvement in scores is evident.

Overall our preliminary results indicate that performance can be improved by:
1. Considering directed graph.
2. Using biased random walks.
3. Embedding nodes in vector space, especially if an apporpriate loss function is used.



![image](https://user-images.githubusercontent.com/71390120/173446782-4f07a794-848b-4fb2-9998-4fd89dc30792.png)

![image](https://user-images.githubusercontent.com/71390120/173681187-da7c20ed-3be6-45e0-944d-d8ecfd006f18.png)

![image](https://user-images.githubusercontent.com/71390120/173446889-58b55ed9-6354-4409-80b0-29c89865cabe.png)

### Further analysis

The embedding in vector space gives us an opportunity to explore the whole network, in relation to the set of seed pages or economic recovery pages.

In particular, vector embeddings can be analysed by clustering methods and dimensionality reduction techniques which enable visual analysis.

To this end, consider mapping graph nodes to a 10-dimensional vector space and using TSNE to reduce the result to two dimensions. In the plot below the axis correspond to TSNE variables and orange dots are economic recovery pages.

While not particularly beautiful, the economic recovery pages seem to exhibit a cluster structure. The TSNE mapping is highly random, and, over repeated replications, only two (rather than three as this plot may suggest) major clusters of ER pages seems to appear regularly - and these are mosttly visible along the y-axis.

![tsne](https://user-images.githubusercontent.com/71390120/173683883-d97ea3f6-696b-43a8-b490-a95262816303.png)

This encouraged us to look further. We use K-means clustering on a 10-dimensional vector space for this purpose.

The numer of clusters is chosen using the sum of squared distances from cluster centres (see the plot below). Base don this metric we opt for 8 clusters.
![image](https://user-images.githubusercontent.com/71390120/173684277-113f9186-ddc7-4022-a868-0ad96da0ce27.png)

Interestingly, most of ER pages (33 out of 40) fall into two clusters, supporting the results of TSNE analysis.

The plot below illustrates the connections between ER pages, with pages colored by cluster (2 pages were omitted as distort the plot). 

![image](https://user-images.githubusercontent.com/71390120/173689496-61414ecb-492a-45cb-a18e-00ad74b6b8dd.png)


It appears that the first cluster (purple) relates to a large extent to education, as it include sfor exmaple pages like:
/topic/further-education-skills/apprenticeships
/browse/working/finding-job
/become-apprentice
/browse/education/find-course
/become-apprentice/apply-for-an-apprenticeship
/find-traineeship
/browse/education

[[[link](https://user-images.githubusercontent.com/71390120/173688686-953878a4-0cf6-41d9-ace1-237f252c8bb8.png)|width=400px]]

The second cluster (orange), on the other hand relates to more general queries, in particular claiming benefits.


## Named-entity recognition 
To enrich the nodes (webpages) from gov.uk with features prior to implementing GNNs, inference was performed from a previously created Named-Entity Recognition (NER) model which utilises the DistillBERT architecture (https://arxiv.org/abs/1910.01108). This model can be used to highlight named-entities in a number of categories (e.g., organisation, people) which can be visualisd as follows: 

<img width="1741" alt="Screenshot 2022-06-16 at 19 19 49" src="https://user-images.githubusercontent.com/104083260/174139092-45e6010a-2462-4e46-b227-d55c6418605a.png">

The NER script in this repo can be used to output data into a .csv file in the following format:
<img width="297" alt="Screenshot 2022-06-16 at 19 21 37" src="https://user-images.githubusercontent.com/104083260/174139369-87196229-ec26-4185-a21c-44ac116470b9.png">

## Graph Neural Networks (GNNs)
The plan is to experiment with a number of GNN architectures (e.g., GAT, Hyperbolic GCN) to make edge level predictions. 
