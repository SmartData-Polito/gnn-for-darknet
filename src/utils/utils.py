import numpy as np
import torch
import os
import shutil
import scipy.sparse as sp

def _normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1)) + 1e-15
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def get_set_diff(A,B):
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return np.array([x for x in aset.difference(bset)])

def compute_accuracy(y_true, y_pred):
    y_pred = torch.round(y_pred)

    return torch.sum(y_pred == y_true) / len(y_true)

def get_diagonal_features(n_nodes):
        features = torch.sparse.FloatTensor(
            indices=torch.stack([
                torch.arange(0, n_nodes),
                torch.arange(0, n_nodes)
            ], dim=0), 
            values=torch.ones([n_nodes, ]),
            size=(n_nodes, n_nodes)
        )

        return features

def initalize_output_folder(folder_name):
    # Remove existing folder
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    
    # Re-make the experiment folder
    os.makedirs(folder_name)
    
    # Initialize training history logfile
    with open(folder_name + "/training_history.csv", "a") as f:
        f.write(f"Epoch,Train_loss,Train_acc")