Explanations here.

Structure:
1. Current approach:
2. - use undirected graph where probability of trnasition is calculated as ...
3. initialise RWs from a (small) set of nodes (seed pages)
4. Calculate pfpf (page-frequency-path-frequency) metric to rank 

We proposed the following changes in the ranking procedure:
1. Directed graph
2. Biased random walks (homophily vs equivariance)
3. Word2vec embedding into vector space

# Performance metric

To compare different ranking algorithms we use the following performance measure.

For a single run of the original ranking algorithm, we produce a list of top 100 pages. These are manually labelled (1 = relevant, 0 = irrelevant) and used for algorithm evaluation ("evaluation set") using the following metric:

$$ \text{score} = \frac{ \text{median}(relevant) - \text{median}(irrelevant)}{ \sigma(relevant) + \sigma(irrelevant) }, $$

where $ relevant $ and $ irrelevant $ are rankings of evaluation set pages labelled 1 and 0 respectively, and $ \sigma $ is a standard deviation.

# Evaluation of proposed changes

To choose which of the proposed steps has positive impact on rankings we ran the following ranking algorithms.

1. **Biased random walks**: Use biased RWs (from the same seeds, using the same walk length, and the same ranking metric (pfpf)). This measures proposed changes 1 & 2.
2. **Biased random walks & vector embeddings**: Use N2V embedding to vector space. Metric is a simple L2 norm distance from the same seeds as original ranking procedure.
3. **Biased random walks & vector embeddings & W2V scores** (this corresponds to Node2Vec approach).


The scores for different parameter combinations are below.

![image](https://user-images.githubusercontent.com/71390120/173446782-4f07a794-848b-4fb2-9998-4fd89dc30792.png)

![image](https://user-images.githubusercontent.com/71390120/173446889-58b55ed9-6354-4409-80b0-29c89865cabe.png)



5. Clustering of ER pages
6. Ranking based on: 
  - Nearest neghibour distances from seeds, ER clusters or all ER pages
  - W2V embedding

Ranking systems run multipel times and mean scores. 

Simulations




1. **Biased RW**: Change RWs (from same seedns, walk length, same ranking metric (pfpf).
2. **NOW**:  **Vector embeddings (similarity metric)**: RWs from all nodes, W2V embedding and N2V similairty measure to score (based on maximum similarity vs seed pages)
3. **To DO**: **Vector embeddings (norm metric)**: for the same graph, create all embeddings, calculate all different KNN scores
