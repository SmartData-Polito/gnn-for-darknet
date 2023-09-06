import numpy as np
from ..utils import get_set_diff
from torch import tensor
from torch import unique as tc_unique
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def generate_negatives(anomaly_num, active_source, active_dest, real_edges):
    """ Generate negative edges for self-supervised training.

    Parameters:
    -----------
    anomaly_num : int
        Number of negative edges to generate.
    active_source : numpy.ndarray
        Array of active source nodes.
    active_dest : numpy.ndarray
        Array of active destination nodes.
    real_edges : numpy.ndarray
        Array of real edges in the graph.

    Returns:
    --------
    torch.Tensor
        A tensor containing the generated negative edges.

    Notes:
    ------
    - This function generates negative edges for self-supervised training.
    - It randomly selects source and destination nodes and ensures they are not in the real edges.
    """
    n_extractions = 10 * anomaly_num
    idx_1 = np.expand_dims(np.transpose(
        np.random.choice(active_source, n_extractions)) , axis=1)
    idx_2 = np.expand_dims(np.transpose(
        np.random.choice(active_dest, n_extractions)) , axis=1)
    fake_edges = np.concatenate((idx_1, idx_2), axis=1).astype(real_edges.dtype)
    
    # Remove duplicates and existing edges
    fake_edges = np.unique(fake_edges, axis=0) 
    fake_edges = get_set_diff(fake_edges, real_edges.T)
    neg_edges = fake_edges[:anomaly_num, :]

    return tensor(neg_edges)

def get_self_supervised_edges(X_to_predict, cuda, ns): 
    """ Get self-supervised edges for training.

    Parameters:
    -----------
    X_to_predict : torch.Tensor
        The input adjacency matrix for which self-supervised edges are generated.
    cuda : bool
        Indicates whether to use CUDA (GPU) for tensor operations.
    ns : int
        Number of negative samples to generate for each positive edge.

    Returns:
    --------
    tuple
        A tuple containing the generated negative edges tensor and the index tensor.

    Notes:
    ------
    - This function generates self-supervised negative edges for training a graph neural network.
    - It filters and samples negative edges based on the input adjacency matrix `X_to_predict`.
    """   
    index = X_to_predict.coalesce().indices()
    filtered_index = index[:, index[0] < index[1]]
    
    # Sample negative edges -- Bipartite graph
    active_sources = tc_unique(filtered_index[0])
    active_dest = tc_unique(filtered_index[1])
    neg_edges = generate_negatives(
        ns * filtered_index.shape[1], 
        active_sources.cpu().numpy(),
        active_dest.cpu().numpy(), 
        index.cpu().numpy()
    ).T
    
    if cuda:
        return neg_edges.cuda(), index.cuda()
    else:    
        return neg_edges, index

def load_single_file(file, day): 
    """ Load and preprocess a single data file.

    Parameters:
    -----------
    file : str
        The path to the data file to load.
    day : int
        The day associated with the loaded data.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing the preprocessed data.

    Notes:
    ------
    - This function loads a data file, filters it to keep only TCP packets, and adds an "interval" column.
    - It also converts string columns to integers for specific fields.
    """   
    # Load the provided file
    df = pd.read_csv(file, sep=' ')
    
    # Keep only TCP packets -- value "6" from tsat
    df = df[df.proto==6].drop(columns=['proto'])
    
    # Add the "interval" column
    df['interval'] = day

    # Covert strings to int
    df[['t_mss', 't_win', 't_ts']] = df[['t_mss', 't_win', 't_ts']]\
                                                .replace({'-':'0'}).astype(int)
    
    return df

def apply_packets_filter(df, min_packets):
    """ Apply a packet count filter to a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing packet data.
    min_packets : int
        The minimum number of packets a source IP must have to be retained.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the packet count filter applied.

    Notes:
    ------
    - This function filters a DataFrame based on the minimum packet count for source IPs.
    """
    # Get daily src IP frequency
    tmp = df.value_counts('src_ip')
    
    # Define and apply the filter
    tmp = tmp[tmp>min_packets].index
    df = df[df.src_ip.isin(tmp)]
    
    return df

def apply_port_filter(df, max_ports):
    """ Apply a port count filter to a DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        The input DataFrame containing packet data.
    max_ports : int
        The maximum number of ports to retain in the "dst_port" column.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with the port count filter applied.

    Notes:
    ------
    - This function filters a DataFrame based on the maximum port count in the "dst_port" column.
    """
    # Get dst ports popularity
    tmp = df.value_counts('dst_port').index
    
    # Define and apply the filter
    port_filter = {p:'oth' for p in tmp[max_ports:]}
    df['dst_port'] = df['dst_port'].map(
        lambda x: port_filter[x] if x in port_filter else x
    )
    
    return df