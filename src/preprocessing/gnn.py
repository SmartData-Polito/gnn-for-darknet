import pandas as pd
import numpy as np
from ..utils import _sparse_mx_to_torch_sparse_tensor, _normalize
from tqdm.notebook import tqdm_notebook as tqdm
from scipy.sparse import coo_matrix

# =============================================================================
# GENERATE GRAPH
# =============================================================================
def extract_single_snapshot(df, day):
    """ Extract and format a single snapshot from a DataFrame for a given day.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.
    day : int
        The specific day for which to extract the snapshot.

    Returns:
    --------
    str
        A formatted string representing the snapshot for the given day.

    Notes:
    ------
    - This function extracts and formats a single snapshot of network data for a given day.
    - It is assumed that the DataFrame `df` has a 'src_ip', 'port', 'weight', and 'label' columns.

    """
    # Extract snapshots
    snapshot = df[df.interval==day]
    snapshot = snapshot.drop(columns=['interval']).values

    # Entries order: src_ip, port, weight, label
    snapshot = [','.join([
        str(x[0]), str(x[1]), str(x[2]), str(x[3])
    ]) for x in snapshot]
    snapshot = '\n'.join(snapshot)
    
    return snapshot

def aggregate_edges(df, gt):
    """ Aggregate edges in a DataFrame while counting packets and adding labels.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.
    gt : pd.DataFrame
        The ground truth labels for source IPs.

    Returns:
    --------
    pd.DataFrame
        An aggregated DataFrame with added packet count and labels for edges.

    Notes:
    ------
    - This function aggregates edges in the DataFrame `df`, counting packets and adding labels.
    - It assumes that the DataFrame `df` has 'src_ip', 'dst_port', and 'interval' columns.
    - The `gt` DataFrame provides ground truth labels for source IPs.

    """
    # Add one column for packets
    df['pkts'] = 1
    
    # Aggregate edges counting (src_ip, dst_ports)
    df = df.groupby(['src_ip', 'dst_port']).agg({
        'interval':lambda x:list(x)[0],
        'pkts':'count'
    }).reset_index()
    
    # Add label column for edges
    df = df.merge(gt, on='src_ip', how='left').fillna('unknown')
    
    return df


# =============================================================================
# TARGETED PORTS BY A SOURCE IP
# =============================================================================
def get_contacted_dst_ports(df):
    """ Get the total number of contacted destination ports per source IP.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the total number of contacted destination ports per source IP.

    Notes:
    ------
    - This function calculates the total number of contacted destination ports per source IP.
    - It assumes that the DataFrame `df` has 'src_ip' and 'dst_port' columns.

    """
    # Total number of contacted ports
    stat = df.groupby('src_ip').agg({'dst_port':lambda x: len(set(x))})
    
    return stat.fillna(.0)

def get_stats_per_dst_port(df):
    """ Get general statistics of packets per destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with general statistics of packets per destination port per source IP.

    Notes:
    ------
    - This function calculates general statistics of packets per destination port per source IP.
    - It assumes that the DataFrame `df` has 'src_ip', 'dst_port', and 'interval' columns.

    """
    # General statistics of packets per destination port
    tmp = df.groupby(['src_ip', 'dst_port'])['interval']\
            .count()\
            .reset_index()\
            .rename(columns={'interval':'pkts'})
    stat = tmp.groupby('src_ip').agg({'pkts':[min, max, sum, 'mean', 'std']})
    
    return stat.fillna(.0)


# =============================================================================
# SOURCE IPS TARGETING A DARKNET PORT
# =============================================================================
def get_contacted_src_ips(df):
    """ Get the total number of contacted source IPs per destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the total number of contacted source IPs per destination port.

    Notes:
    ------
    - This function calculates the total number of contacted source IPs per destination port.
    - It assumes that the DataFrame `df` has 'src_ip' and 'dst_port' columns.
    
    """
    # Total number of contacted ports
    stat = df.groupby('dst_port').agg({'src_ip':lambda x: len(set(x))})
    
    return stat.fillna(.0)

def get_stats_per_src_ip(df):
    """ Get general statistics of packets per source IP per destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with general statistics of packets per source IP per destination port.

    Notes:
    ------
    - This function calculates general statistics of packets per source IP per destination port.
    - It assumes that the DataFrame `df` has 'src_ip', 'dst_port', and 'interval' columns.
    """
    # General statistics of packets per destination port
    tmp = df.groupby(['src_ip', 'dst_port'])['interval']\
            .count()\
            .reset_index()\
            .rename(columns={'interval':'pkts'})
    stat = tmp.groupby('dst_port').agg({'pkts':[min, max, sum, 'mean', 'std']})
    
    return stat.fillna(.0)


# =============================================================================
# DARKNET IPS TARGETED BY A SOURCE IP
# =============================================================================
def get_contacted_dst_ips(df, dummy=False):
    """ Get the total number of contacted darknet IPs per source IP or destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.
    dummy : bool, optional
        If True, calculates the total number of contacted darknet IPs per destination port, 
        by default False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the total number of contacted darknet IPs per source IP or 
        destination port.

    Notes:
    ------
    - This function calculates the total number of contacted darknet IPs per source IP or 
      destination port.
    - If `dummy` is True, it calculates the total number of contacted darknet IPs per 
      destination port.
    - It assumes that the DataFrame `df` has 'src_ip', 'dst_ip', and 'dst_port' columns.
    """
    # Total number of contacted darknet IPs
    if not dummy:
        stat = df.groupby('src_ip').agg({'dst_ip':lambda x: len(set(x))})
    
    else:
        stat = df.groupby('dst_port').agg({'dst_ip':lambda x: len(set(x))})
        for col in stat.columns:
            stat[col].values[:] = 0
    
    return stat.fillna(.0)

def get_stats_per_dst_ip(df, dummy=False):
    """ Get general statistics of packets per destination IP per source IP or destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.
    dummy : bool, optional
        If True, calculates statistics per destination IP per destination port, 
        by default False.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with general statistics of packets per destination IP per source IP or 
        destination port.

    Notes:
    ------
    - This function calculates general statistics of packets per destination IP per source 
      IP or destination port.
    - If `dummy` is True, it calculates statistics per destination IP per destination port.
    - It assumes that the DataFrame `df` has 'src_ip', 'dst_ip', 'dst_port', and 
      'interval' columns.
    """
    # General statistics of packets per destination ip
    if not dummy:
        tmp = df.groupby(['src_ip', 'dst_ip'])['interval']\
                .count()\
                .reset_index()\
                .rename(columns={'interval':'pkts'})

        stat = tmp.groupby('src_ip').agg({'pkts':[min, max, sum, 'mean', 'std']})
    else:
        tmp = df.groupby(['dst_port', 'dst_ip'])['interval']\
                .count()\
                .reset_index()\
                .rename(columns={'interval':'pkts'})

        stat = tmp.groupby('dst_port').agg({'pkts':[min, max, sum, 'mean', 'std']})
        
        for col in stat.columns:
            stat[col].values[:] = 0

    return stat.fillna(.0)

# =============================================================================
# GENERIC PACKETS STATISTICS
# =============================================================================
def get_packet_statistics(df, by='src_ip'):
    """ Get general packet statistics per source IP or destination port.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing network data.
    by : str, optional
        The column by which to group the packet statistics ('src_ip' or 'dst_port'), 
        by default 'src_ip'.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with general packet statistics per source IP or destination port.

    Notes:
    ------
    - This function calculates general packet statistics per source IP or destination port.
    - The `by` parameter specifies whether to group the statistics by source IP or 
      destination port.
    - It assumes that the DataFrame `df` contains relevant columns for packet statistics.
    """
    # General packets statistics
    stat = df.groupby(by).agg({
        'pck_len':[sum, min, max, 'mean', 'std'],
        'ttl':[sum, min, max, 'mean', 'std'],
        't_mss':[sum, min, max, 'mean', 'std'],
        't_win':[sum, min, max, 'mean', 'std'],
        't_ts':[sum, min, max, 'mean', 'std']
    })
    
    return stat.fillna(.0)


def uniform_features(df, lookup, node_type):
    """ Uniformly format and index features DataFrame based on node lookup.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing node features.
    lookup : dict
        A dictionary mapping node names to IDs.
    node_type : str
        The type of nodes in the DataFrame (e.g., 'src_ip', 'dst_port').

    Returns:
    --------
    pd.DataFrame
        A uniformly formatted and indexed DataFrame of node features.

    Notes:
    ------
    - This function uniformly formats and indexes a DataFrame of node features based on a 
      node lookup dictionary.
    - It assumes that the DataFrame `df` has a column named as specified by `node_type`.
    """
    # Concatenate and normalize
    df = pd.concat(df, axis=1)
    df = df/df.max()
    
    # Uniform single features dataframe
    df.columns = range(df.shape[1])
    df = df.reindex(lookup.keys()).fillna(.0).reset_index()
    
    # Replace nodes with node ID
    df[node_type] = df[node_type].map(lambda x: lookup[x])
    df = df.rename(columns={node_type:'index'}).set_index('index')
    
    return df



def generate_adjacency_matrices(flist, weighted=True):
    """ Generate adjacency matrices from a list of DataFrame files.

    Parameters:
    -----------
    flist : list
        A list of file paths, each containing a DataFrame of network data.
    weighted : bool, optional
        If True, the edges in the generated matrices will be weighted, 
        by default True.

    Returns:
    --------
    list
        A list of torch sparse tensors representing the adjacency matrices.

    Notes:
    ------
    - This function generates adjacency matrices from a list of DataFrame files.
    - If `weighted` is True, the edges in the matrices will be weighted based on packet 
      counts.
    """
    traces = []
    edges = []
    for fname in tqdm(flist, desc='Loading graphs'):
        data = pd.read_csv(fname, header=None, 
                           names=["src", "dst", "weight", "label"])
        if not weighted:
            data['weight'] = 1

        self_loops_value = 1
        traces.append(data)

        edges.append(data[['src', 'dst', 'weight']])

    tot_nodes = max([max(edge.src.max(), edge.dst.max()) for edge in edges])+1

    indices = [np.stack([
        # Source nodes
        np.concatenate((
            edge.src.values, # Original edges
            edge.dst.values, # Symmetric edges
            np.arange(tot_nodes) # Self-loops
        )),
        # Destination nodes
        np.concatenate((
            edge.dst.values, # Original edges
            edge.src.values, # Symmetric edges
            np.arange(tot_nodes) # Self-loops
        ))]) for edge in edges]

    values = [
        np.concatenate((
            edge["weight"].values.reshape(-1,), 
            edge["weight"].values.reshape(-1,), 
            self_loops_value*np.ones([tot_nodes,])
        ))
        for edge in edges]
    
    adjs = []
    for i in tqdm(range(len(edges)), desc='Generating matrices'):
        adjs.append(
            _sparse_mx_to_torch_sparse_tensor(
                _normalize(
                    coo_matrix(
                        (values[i], 
                         (indices[i][0], indices[i][1])), 
                        shape=(tot_nodes, tot_nodes),
                        dtype=np.float32))))
    
    return adjs