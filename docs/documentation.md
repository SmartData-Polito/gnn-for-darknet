# Exploring Temporal GNN Embeddings for Darknet Traffic Analysis
## API Reference
This is the class and function reference of library used in this repo. 

### [`src.models.classification`](classification.md): Classifiers
Class implementing the k-Nearest-Neighbors classifier

- [`src.models.classification.KnnClassifier`](classification.md): This class defines the KNN Classifier with specified hyperparameters 
and creates a StandardScaler for data standardization. If `_load_model` 
        is True, it loads a pre-trained model and scaler from the specified path.

___

### [`src.models.gnn`](gnn.md): GNN embeddings

- [`src.models.gnn.GCN`](gnn.md#srcmodelsgnngcn): description
- [`src.models.gnn.GCN_GRU`](gnn.md#srcmodelsgnngcn_gru): description
- [`src.models.gnn.IncrementalGcnGru`](gnn.md#srcmodelsgnnincrementalgcngru): description

___


### [`src.models.nlp`](nlp.md): NLP embeddings

- [`src.models.nlp.iWord2Vec`](nlp.md): description

___


### [`src.preprocessing`]: Preprocessing Functions

- `src.preprocessing.gnn.extract_single_snapshot`
- `src.preprocessing.gnn.aggregate_edges`
- `src.preprocessing.gnn.get_contacted_dst_ports`
- `src.preprocessing.gnn.get_stats_per_dst_port`
- `src.preprocessing.gnn.get_contacted_src_ips`
- `src.preprocessing.gnn.get_stats_per_src_ip`
- `src.preprocessing.gnn.get_contacted_dst_ips`
- `src.preprocessing.gnn.get_stats_per_dst_ip`
- `src.preprocessing.gnn.get_packet_statistics`
- `src.preprocessing.gnn.uniform_features`
- `src.preprocessing.gnn.generate_adjacency_matrices`

- `src.preprocessing.nlp.drop_duplicates`
- `src.preprocessing.nlp.split_array`

- `src.preprocessing.preprocessing.generate_negatives`
- `src.preprocessing.preprocessing.get_self_supervised_edges`

___


### [`src.utils`]: Utility Functions