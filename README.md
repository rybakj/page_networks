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

To choose which of these steps has positive impact on rankings we use the following measure.

For a single run of the original ranking algorithm, we produce a list of top 100 pages. These are manually labelled (1 = relevant, 0 = irrelevant) and used for algorithm evaluation using the following metric:

$$ score = \frac{ median(relevant) - median(irrelevant)}{ \sigma(relevant) + \sigma(irrelevant) } $$

5. Clustering of ER pages
6. Ranking based on: 
  - Nearest neghibour distances from seeds, ER clusters or all ER pages
  - W2V embedding

Ranking systems run multipel times and mean scores. 

Simulations




1. **Biased RW**: Change RWs (from same seedns, walk length, same ranking metric (pfpf).
2. **NOW**:  **Vector embeddings (similarity metric)**: RWs from all nodes, W2V embedding and N2V similairty measure to score (based on maximum similarity vs seed pages)
3. **To DO**: **Vector embeddings (norm metric)**: for the same graph, create all embeddings, calculate all different KNN scores
