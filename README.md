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
- The `requirements.txt` file lists the required Python packages and their versions.

### Notebook Folder

The `notebook` folder contains Jupyter notebooks that demonstrate how to reproduce the experiments described in the paper. Each notebook corresponds to a specific experiment and provides step-by-step instructions and explanations. The notebooks are designed to be self-contained and easy to follow.

- The notebook [`00_dataset_characterization`](notebooks/00_dataset_characterization.ipynb) contains the main codes to characterized both the filtered dataset (total and on daily basis) and the resulting temporal graph.

- The notebook [`01_dataset_generation`](notebooks/01_dataset_generation.ipynb) contains the main codes to (i) process raw traces filtering unwanted data; (ii) generate bipartite graphs from filtered traces and extract node features; (iii) generate textual corpora which will be processed by NLP algorithms.

- The notebook [`02_embeddings_generation`](notebooks/02_embeddings_generation.ipynb) contains the main codes to (i) prduce NLP embeddings through i-DarkVec; (ii) prduce (t)GNN embeddings without node features; (iii) prduce (t)GNN embeddings with node features; (iv) produce embeddings to evaluate the impact of the parameters (history and training epochs).

- The notebook [`03_classification`](notebooks/03_classification.ipynb) contains the main codes to run the final k-Nearest-Neighbors classification pipeline. The main experiments are: (i) Main table with classification performance; (ii) impact of History parameter -- temporal aspect of tGNN; (iii) Impact of training epochs for incremental training.

## Data Structure


## API Reference

Please, refer to the [API reference](docs/documentation.md) for the complete code documentation.


# ToDo
- Description of data folder
- Add GCN (early stop)
- Upload data