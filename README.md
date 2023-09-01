# Exploring Temporal GNN Embeddings for Darknet Traffic Analysis

This repo contains the source codes for the paper "The Benefit of GNNs for Network Traffic Analysis" submittet to the 2nd International Workshop on Graph Neural Networking (GNNet@CoNEXT’23).

## Table of Contents

- [Repository Structure](#repository-structure)
    - [Notebook Folder](#notebook-folder)
    - [Src Folder](#library-folder)
- [Installation](#installation)

## Repository Structure

The repository is organized as follows:

- The `notebook` folder contains Jupyter notebooks that replicate the experiments presented in the paper.
- The `src` folder includes the Python library that provides functions, methods, classes, and models used in the notebooks.
- The `requirements.txt` file lists the required Python packages and their versions.

### Notebook Folder

The `notebook` folder contains Jupyter notebooks that demonstrate how to reproduce the experiments described in the paper. Each notebook corresponds to a specific experiment and provides step-by-step instructions and explanations. The notebooks are designed to be self-contained and easy to follow.

#### Preprocessing stages

The notebook `00_dataset_characterization` contains the main codes to (i) process raw traces filtering unwanted data; (ii) generate bipartite graphs from filtered traces and extract node features; (iii) generate textual corpora which will be processed by NLP algorithms.

The notebook `01_dataset_generation` contains the main codes to (i) process raw traces filtering unwanted data; (ii) generate bipartite graphs from filtered traces and extract node features; (iii) generate textual corpora which will be processed by NLP algorithms.


### Src Folder

The `src` folder houses a Python library that provides the necessary tools for implementing and reproducing the experiments. This library encapsulates the functions, methods, classes, and models used in the notebooks. By utilizing this library, users can streamline their workflow and easily experiment with different components.

## Installation

To set up the required environment, you can use the provided `requirements.txt` file. Run the following command to install the necessary dependencies using pip:

```bash
pip3 install -r requirements.txt
```

Feel free to explore the notebooks, experiment with the library, and adapt the code to your own research or applications.

For any questions, issues, or suggestions, please open an issue on this repository.