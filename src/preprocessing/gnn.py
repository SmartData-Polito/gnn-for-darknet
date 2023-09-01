import pandas as pd
import numpy as np
from ..utils import _sparse_mx_to_torch_sparse_tensor, _normalize
from tqdm.notebook import tqdm_notebook as tqdm
from scipy.sparse import coo_matrix

# =============================================================================
# GENERATE GRAPH
# =============================================================================
def extract_single_snapshot(df, day):
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
    # Total number of contacted ports
    stat = df.groupby('src_ip').agg({'dst_port':lambda x: len(set(x))})
    
    return stat.fillna(.0)

def get_stats_per_dst_port(df):
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
    # Total number of contacted ports
    stat = df.groupby('dst_port').agg({'src_ip':lambda x: len(set(x))})
    
    return stat.fillna(.0)

def get_stats_per_src_ip(df):
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
    # Total number of contacted darknet IPs
    if not dummy:
        stat = df.groupby('src_ip').agg({'dst_ip':lambda x: len(set(x))})
    
    else:
        stat = df.groupby('dst_port').agg({'dst_ip':lambda x: len(set(x))})
        for col in stat.columns:
            stat[col].values[:] = 0
    
    return stat.fillna(.0)

def get_stats_per_dst_ip(df, dummy=False):
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