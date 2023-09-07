# Exploring Temporal GNN Embeddings for Darknet Traffic Analysis

This repo contains the source codes for the paper "The Benefit of GNNs for Network Traffic Analysis" submitted to the 2nd International Workshop on Graph Neural Networking (GNNet@CoNEXT’23).

Feel free to explore the notebooks, experiment with the library, and adapt the code to your own research or applications.
For any questions, issues, or suggestions, please open an issue on this repository.

## Table of Contents
- [How to reproduce results in the paper?](#how-to-reproduce-results-in-the-paper)
- [Repository Structure](#repository-structure)
- [Notebook Folder](#notebook-folder)
- [Data Structure](#data-structure)
- [API Reference](#api-reference)


## How to reproduce results in the paper?
Note: This guide assumes a Debian-like system (tested on Ubuntu 20.04 & Debian 11).

1. Clone this repository
2. Download the gzip data file from: https://TBD
3. Unzip the TBD file into a subfolder of this repository called `data`

```bash
tar -zxvf TBD.tar.gz
```

4. Install the `virtualenv` library (python3 is assumed):

```bash
pip3 install --user virtualenv
```

5. Create a new virtual environment and activate it:

```bash
virtualenv darknet-gnn-env
source darknet-gnn-env/bin/activate
```

6. Install the required libraries (python3 is assumed):

```bash
pip3 install -r requirements.txt
```

7. Run the notebooks described next. For example, to run the first notebook:

```bash
jupyter-lab 00-dataset-characterization.ipynb
```

8. When the notebook exploration is ended, remember to deactivate the virtual environment:

```bash
deactivate
```

## Repository Structure

The repository is organized as follows:

- The `notebook` folder contains Jupyter notebooks that replicate the experiments presented in the paper.
- The `src` folder contains all the source codes and libraries providing the necessary tools for implementing and reproducing the experiments of the paper. This library encapsulates the functions, methods, classes, and models used in the notebooks. By utilizing this library, users can streamline their workflow and easily experiment with different components.
- The `docs` folder contains the codes documentation.
- The `requirements.txt` file lists the required Python packages and their versions.

### Notebook Folder

The `notebook` folder contains Jupyter notebooks that demonstrate how to reproduce the experiments described in the paper. Each notebook corresponds to a specific experiment and provides step-by-step instructions and explanations. The notebooks are designed to be self-contained and easy to follow.

- The notebook [`00_dataset_characterization`](notebooks/00_dataset_characterization.ipynb) contains the main codes to characterized both the filtered dataset (total and on daily basis) and the resulting temporal graph.

- The notebook [`01_dataset_generation`](notebooks/01_dataset_generation.ipynb) contains the main codes to (i) process raw traces filtering unwanted data; (ii) generate bipartite graphs from filtered traces and extract node features; (iii) generate textual corpora which will be processed by NLP algorithms.

- The notebook [`02_embeddings_generation`](notebooks/02_embeddings_generation.ipynb) contains the main codes to (i) prduce NLP embeddings through i-DarkVec; (ii) prduce (t)GNN embeddings without node features; (iii) prduce (t)GNN embeddings with node features; (iv) produce embeddings to evaluate the impact of the parameters (history and training epochs).

- The notebook [`03_classification`](notebooks/03_classification.ipynb) contains the main codes to run the final k-Nearest-Neighbors classification pipeline. The main experiments are: (i) Main table with classification performance; (ii) impact of History parameter -- temporal aspect of tGNN; (iii) Impact of training epochs for incremental training.

## Data Structure

- `corpus` folder contains the NLP corpora. Each pickle file is contains a list of numpy arrays (sequence of strings) of a snapshot named as `corpus_DATE.pkl`, where `DATE` is referred to the considered snapshot. 
- `features` folder contains the node features. Each csv file has _V_ rows, where _V_ is the number of vertices of the graph and _F_ columns, where _F_ is the number of features for each node. Each file is named `features_DATE.csv`, where `DATE` is referred to the considered snapshot.
- `graph` folder contains the graph obtained for each snapshot. Each txt file contains 4 columns (_source node_, _destination node_, _edge weight_, _label_).  Each file is named `DATE.txt`, where `DATE` is referred to the considered snapshot.
- `ground_truth` folder contains the full ground truth. The file `ground_truth.csv` has two columns (_src_ip_, _label_). The first column contains source IP addresses, the second column is the ground truth label.
- `gnn_embeddings` folder contains the csv of the embeddings generated through GNNs. 
    - The files containing embeddings generated without node features are named `embeddings_MODEL_DATE.csv`, where `DATE` is referred to the considered snapshot and `MODEL` is referred to the used GNN.
    - The files containing embeddings generated with node features are named `embeddings_MODEL_features_DATE.csv`, where `DATE` is referred to the considered snapshot and `MODEL` is referred to the used GNN.
    - The files containing embeddings generated for the history evaluation are named `embeddings_MODEL_features_Hhist_DATE.csv`, where `DATE` is referred to the considered snapshot, `MODEL` is referred to the used GNN and `hist` is the value of the history parameter.
    - The files containing embeddings generated for the training evaluation are named `embeddings_MODEL_features_eeEPOCHS_DATE.csv`, where `DATE` is referred to the considered snapshot, `MODEL` is referred to the used GNN and `EPOCHS` is the value of the training epochs.
    
The possible values of `MODEL` are `gcn`, `gcngru`, `igcn`, `igcngru`. 
Each file is indexed by the src_ip active in the considered snapshot and hse _E_ columns, where _E_ is the embeddings size.
- `nlp_embeddings` folder contains the csv of the embeddings generated through i-DarkVec. Each file is named `embeddings_idarkvec_DATE.csv`, where `DATE` is referred to the considered snapshot. Each file is indexed by the src_ip active in the considered snapshot and hse _E_ columns, where _E_ is the embeddings size.
- `raw` folder contains the raw data pre-processed with the codes reported in `01_dataset_generation`. Each csv file contains the following columns (ts, ethtype, src_ip, src_port, dst_ip, dst_port, pck_len, tcp_flags, mirai, tcp_seq, ttl, t_mss, t_win, t_ts, t_sack, t_sackp, interval). Each file is named `raw_DATE.csv`, where `DATE` is referred to the considered snapshot.
- `traces` folder contains the pure raw traces. Each file is named `trace_YYMMDD_HH_MM_SS_MS.log.gz` and each row is referred to a packet received by the darknet.
- `results` folder contains the final results to be plotted.

## API Reference

Please, refer to the [API reference](docs/documentation.md) for the complete code documentation.

## ToDo
- Add GCN (early stop)
- Upload data