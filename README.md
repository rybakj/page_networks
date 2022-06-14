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

All of the apporaches discussed here involve some randomness, predominantly coming from random walks. For this reason, scores given below are averaged over multiple runs (usually 10 or 20) of the same method for the same combination of hyperparameters. 

# Evaluation of proposed changes

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

# Further analysis

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


# Notes for self

5. Clustering of ER pages
6. Ranking based on: 
  - Nearest neghibour distances from seeds, ER clusters or all ER pages
  - W2V embedding

Ranking systems run multiple times and mean scores. 

Simulations




1. **Biased RW**: Change RWs (from same seedns, walk length, same ranking metric (pfpf).
2. **NOW**:  **Vector embeddings (similarity metric)**: RWs from all nodes, W2V embedding and N2V similairty measure to score (based on maximum similarity vs seed pages)
3. **To DO**: **Vector embeddings (norm metric)**: for the same graph, create all embeddings, calculate all different KNN scores
