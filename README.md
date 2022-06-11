Explanations here.

Structure:
1. Current approach:
2. - use undirected graph where probability of trnasition is calculated as ...
3. initialise RWs from a (small) set of nodes (seed pages)
4. Calculate tfdf metric to rank 

Changes
1. Directed graph
2. Biased random walks (homophily vs equivariance)
3. Word2vec embedding into vector space
4. Clustering of ER pages
5. Ranking based on: 
  - Nearest neghibour distances from seeds, ER clusters or all ER pages
  - W2V embedding

Ranking systems run multipel times and mean scores. 
