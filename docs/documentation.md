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
Class implementing Graph Neural Networks to generate embeddings in a self-supervised way.

- [`src.models.gnn.GCN`](gnn.md#srcmodelsgnngcn): This class implements the GCN (Graph Convolutional Network) model 
        with the specified hyperparameters.
- [`src.models.gnn.GCN_GRU`](gnn.md#srcmodelsgnngcn_gru): This class implements the GCN-GRU model with the specified hyperparameters.
        It sets up the model, optimizer, and manages CUDA if available and specified.
- [`src.models.gnn.IncrementalGcnGru`](gnn.md#srcmodelsgnnincrementalgcngru): This class implements the Incremental GCN-GRU model with the specified
        hyperparameters. It sets up the model, optimizer, and manages CUDA if available
        and specified.

___


### [`src.models.nlp`](nlp.md): NLP embeddings
Class implementing i-DarkVec to generate embeddings in a self-supervised way.

- [`src.models.nlp.iWord2Vec`](nlp.md): This class implements a iWord2Vec model.

___


### [`src.preprocessing`](preprocessing.md): Preprocessing Functions
Preprocessing functions used to generate GNN embeddings

- `src.preprocessing.gnn.extract_single_snapshot`: Extract and format a single snapshot from a DataFrame for a given day.
- `src.preprocessing.gnn.aggregate_edges`: Aggregate edges in a DataFrame while counting packets and adding labels.
- `src.preprocessing.gnn.get_contacted_dst_ports`: Get the total number of contacted destination ports per source IP.
- `src.preprocessing.gnn.get_stats_per_dst_port`: Get general statistics of packets per destination port.
- `src.preprocessing.gnn.get_contacted_src_ips`: Get the total number of contacted source IPs per destination port.
- `src.preprocessing.gnn.get_stats_per_src_ip`: Get general statistics of packets per source IP per destination port.
- `src.preprocessing.gnn.get_contacted_dst_ips`: Get the total number of contacted darknet IPs per source IP or destination port.
- `src.preprocessing.gnn.get_stats_per_dst_ip`: Get general statistics of packets per destination IP per source IP or destination port.
- `src.preprocessing.gnn.get_packet_statistics`: Get general packet statistics per source IP or destination port.
- `src.preprocessing.gnn.uniform_features`: Uniformly format and index features DataFrame based on node lookup.
- `src.preprocessing.gnn.generate_adjacency_matrices`: Generate adjacency matrices from a list of DataFrame files.

Preprocessing functions used to generate NLP embeddings

- `src.preprocessing.nlp.drop_duplicates`: Remove consecutive duplicate elements from a NumPy array.
- `src.preprocessing.nlp.split_array`: Split a NumPy array into smaller sub-arrays of a specified step size.

Generic preprocessing functions
- `src.preprocessing.preprocessing.generate_negatives`: Generate negative edges for self-supervised training.
- `src.preprocessing.preprocessing.get_self_supervised_edges`: Get self-supervised edges for training.
- `src.preprocessing.preprocessing.load_single_file`: Load and preprocess a single data file.
- `src.preprocessing.preprocessing.apply_packets_filter`: Apply a packet count filter to a DataFrame.
- `src.preprocessing.preprocessing.apply_port_filter` : Apply a port count filter to a DataFrame.


___


### [`src.utils`](utils.md): Utility Functions
Generic utility functions

- `src.utils._normalize`: Row-normalize a sparse matrix.
- `src.utils._sparse_mx_to_torch_sparse_tensor`: Convert a scipy sparse matrix to a torch sparse tensor.
- `src.utils.get_set_diff`: Compute the set difference between two arrays A and B.
- `src.utils.compute_accuracy`: Compute accuracy between true and predicted labels.
- `src.utils.get_diagonal_features`: Get a sparse diagonal feature matrix.
- `src.utils.initalize_output_folder`: Initialize an output folder for experiment results.