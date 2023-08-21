import numpy as np
from ..utils import get_set_diff
from torch import tensor
from torch import unique as tc_unique
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def generate_negatives(anomaly_num, active_source, active_dest, real_edges):
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
    # Get daily src IP frequency
    tmp = df.value_counts('src_ip')
    
    # Define and apply the filter
    tmp = tmp[tmp>min_packets].index
    df = df[df.src_ip.isin(tmp)]
    
    return df

def apply_port_filter(df, max_ports):
    # Get dst ports popularity
    tmp = df.value_counts('dst_port').index
    
    # Define and apply the filter
    port_filter = {p:'oth' for p in tmp[max_ports:]}
    df['dst_port'] = df['dst_port'].map(
        lambda x: port_filter[x] if x in port_filter else x
    )
    
    return df