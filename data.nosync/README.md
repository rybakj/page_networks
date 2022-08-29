
Due to the large size of data files the "data" folder is hosted on Google Drive:

https://drive.google.com/drive/folders/1rw2TIXTx5Fd9umHaXrYCaX5NCQ94tCSj?usp=sharing

Subfolder structure:
- `inputs`: Inputs include NER data, graph from BigQuery run, the labelled data and the top 40 most relevant pages selected by GDS team using the original method (though this file is not used in the original analysis).
- `outputs`: Outputs from inidividual methods and graph processing
(The above two folders are used mostly in graph processing, context_approaches notebooks and unsupervised GNN.)
- `ner_and_sgnn`: Files used in NER analysis and semi-supervised GNN. 