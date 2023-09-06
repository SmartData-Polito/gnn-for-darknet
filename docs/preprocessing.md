# `src.preprocessing`

```python
extract_single_snapshot(df, day)
```

[source](../src/preprocessing/gnn.py)

Extract and format a single snapshot from a DataFrame for a given day.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.
- **day**:_int_

    The specific day for which to extract the snapshot.

#### Returns
- _str_

    A formatted string representing the snapshot for the given day.
    
___

```python
aggregate_edges(df, gt)
```
[source](../src/preprocessing/gnn.py)

Aggregate edges in a DataFrame while counting packets and adding labels.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.
- **gt**:_pd.DataFrame_

    The ground truth labels for source IPs.

#### Returns
- _pd.DataFrame_

    An aggregated DataFrame with added packet count and labels for edges.
___

```python
get_contacted_dst_ports(df)
```
[source](../src/preprocessing/gnn.py)

Get the total number of contacted destination ports per source IP.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.

#### Returns
- _pd.DataFrame_

    A DataFrame with the total number of contacted destination ports per source IP.

___

```python
get_stats_per_dst_port(df)
```
[source](../src/preprocessing/gnn.py)

Get general statistics of packets per destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.

#### Returns
- _pd.DataFrame_

    A DataFrame with general statistics of packets per destination port per source IP.
___

```python
get_contacted_src_ips(df)
```
[source](../src/preprocessing/gnn.py)

Get the total number of contacted source IPs per destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.

#### Returns
- _pd.DataFrame_

    A DataFrame with the total number of contacted source IPs per destination port.
    
___

```python
get_stats_per_src_ip(df)
```
[source](../src/preprocessing/gnn.py)

Get general statistics of packets per source IP per destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.

#### Returns
- _pd.DataFrame_

    A DataFrame with general statistics of packets per source IP per destination port.
    
___

```python
get_contacted_dst_ips(df, dummy=False)
```
[source](../src/preprocessing/gnn.py)


Get the total number of contacted darknet IPs per source IP or destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.
- **dummy**:_bool, optional_

    If True, calculates the total number of contacted darknet IPs per destination port, 
    by default False.

#### Returns
- _pd.DataFrame_

    A DataFrame with the total number of contacted darknet IPs per source IP or 
    destination port.
___

```python
get_stats_per_dst_ip(df, dummy=False)
```
[source](../src/preprocessing/gnn.py)


Get general statistics of packets per destination IP per source IP or destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.
- **dummy**:_bool, optional_

    If True, calculates statistics per destination IP per destination port, 
    by default False.

#### Returns
- _pd.DataFrame_

    A DataFrame with general statistics of packets per destination IP per source IP or 
    destination port.
___

```python
get_packet_statistics(df, by='src_ip')
```
[source](../src/preprocessing/gnn.py)


Get general packet statistics per source IP or destination port.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing network data.
- **by**:_str, optional_

    The column by which to group the packet statistics ('src_ip' or 'dst_port'), 
    by default 'src_ip'.

#### Returns
- _pd.DataFrame_

    A DataFrame with general packet statistics per source IP or destination port.
___

```python
uniform_features(df, lookup, node_type)
```
[source](../src/preprocessing/gnn.py)


Uniformly format and index features DataFrame based on node lookup.

#### Parameters
- **df**:_pd.DataFrame_

    The input DataFrame containing node features.
- **lookup**:_dict_

    A dictionary mapping node names to IDs.
- **node_type**:_str_

    The type of nodes in the DataFrame (e.g., 'src_ip', 'dst_port').

#### Returns
- _pd.DataFrame_

    A uniformly formatted and indexed DataFrame of node features.

___

```python
generate_adjacency_matrices(flist, weighted=True)
```
[source](../src/preprocessing/gnn.py)


Generate adjacency matrices from a list of DataFrame files.

#### Parameters
- **flist**:_list_

    A list of file paths, each containing a DataFrame of network data.
- **weighted**:_bool, optional_

    If True, the edges in the generated matrices will be weighted, 
    by default True.

#### Returns
- _list_

    A list of torch sparse tensors representing the adjacency matrices.

___

```python
drop_duplicates(x)
```
[source](../src/preprocessing/nlp.py)

Remove consecutive duplicate elements from a NumPy array.

#### Parameters
- **x**:_numpy.ndarray_

    The input NumPy array from which consecutive duplicates will be removed.

#### Returns
- _numpy.ndarray_

    A NumPy array with consecutive duplicate elements removed.

___

```python
split_array(arr, step=1000)
```
[source](../src/preprocessing/nlp.py)

Split a NumPy array into smaller sub-arrays of a specified step size.

#### Parameters
- **arr**:_numpy.ndarray_

    The input NumPy array to be split.
- **step**:_int, optional_

    The size of each sub-array, by default 1000.

#### Returns
- _list_

    A list of NumPy sub-arrays obtained by splitting the input array.

___

```python
generate_negatives(anomaly_num, active_source, active_dest, real_edges)
```
[source](../src/preprocessing/preprocessing.py)

Generate negative edges for self-supervised training.

#### Parameters
- **anomaly_num**:_int_

    Number of negative edges to generate.
- **active_source**:_numpy.ndarray_

    Array of active source nodes.
- **active_dest**:_numpy.ndarray_

    Array of active destination nodes.
- **real_edges**:_numpy.ndarray_

    Array of real edges in the graph.

#### Returns
- _torch.Tensor_

    A tensor containing the generated negative edges.

___

```python
get_self_supervised_edges(X_to_predict, cuda, ns)
```
[source](../src/preprocessing/preprocessing.py)

Get self-supervised edges for training.

#### Parameters
- **X_to_predict**:_torch.Tensor_

    The input adjacency matrix for which self-supervised edges are generated.
- **cuda**:_bool_

    Indicates whether to use CUDA (GPU) for tensor operations.
- **ns**:_int_

    Number of negative samples to generate for each positive edge.

#### Returns
- _tuple_

    A tuple containing the generated negative edges tensor and the index tensor.
    
___

```python
load_single_file(file, day)
```
[source](../src/preprocessing/preprocessing.py)

Load and preprocess a single data file.

#### Parameters
- **file**:_str_

    The path to the data file to load.
- **day**:_int_

    The day associated with the loaded data.

#### Returns
- _pandas.DataFrame_

    A DataFrame containing the preprocessed data.

___

```python
apply_packets_filter(df, min_packets)
```
[source](../src/preprocessing/preprocessing.py)

Apply a packet count filter to a DataFrame.

#### Parameters
- **df**:_pandas.DataFrame_

    The input DataFrame containing packet data.
- **min_packets**:_int_

    The minimum number of packets a source IP must have to be retained.

#### Returns
- _pandas.DataFrame_

    A DataFrame with the packet count filter applied.
    
___


```python
apply_port_filter(df, max_ports)
```
[source](../src/preprocessing/preprocessing.py)

Apply a port count filter to a DataFrame.

#### Parameters
- **df**:_pandas.DataFrame_

    The input DataFrame containing packet data.
- **max_ports**:_int_

    The maximum number of ports to retain in the "dst_port" column.

#### Returns
- _pandas.DataFrame_

    A DataFrame with the port count filter applied.