import numpy as np
import torch
import os
import shutil
import scipy.sparse as sp

def _normalize(mx):
    """ Row-normalize a sparse matrix.

    Parameters:
    -----------
    mx : sp.csr_matrix
        The input sparse matrix.

    Returns:
    --------
    sp.csr_matrix
        The row-normalized sparse matrix.

    Notes:
    ------
    - This function row-normalizes a sparse matrix by dividing each row by its sum.
    - Avoids division by zero by adding a small epsilon (1e-15) to the denominator.
    """
    rowsum = np.array(mx.sum(1)) + 1e-15
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def _sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """ Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters:
    -----------
    sparse_mx : sp.csr_matrix
        The input scipy sparse matrix.

    Returns:
    --------
    torch.sparse.FloatTensor
        The torch sparse tensor equivalent of the input matrix.

    Notes:
    ------
    - This function converts a scipy sparse matrix into a corresponding torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)

    return torch.sparse.FloatTensor(indices, values, shape)

def get_set_diff(A,B):
    """ Compute the set difference between two arrays A and B.

    Parameters:
    -----------
    A : np.ndarray
        The first input array.
    B : np.ndarray
        The second input array.

    Returns:
    --------
    np.ndarray
        An array containing the set difference between A and B.

    Notes:
    ------
    - This function calculates the set difference between two arrays A and B.
    """
    aset = set([tuple(x) for x in A])
    bset = set([tuple(x) for x in B])
    return np.array([x for x in aset.difference(bset)])

def compute_accuracy(y_true, y_pred):
    """ Compute accuracy between true and predicted labels.

    Parameters:
    -----------
    y_true : torch.Tensor
        The true labels.
    y_pred : torch.Tensor
        The predicted labels.

    Returns:
    --------
    torch.Tensor
        The accuracy score.

    Notes:
    ------
    - This function computes the accuracy score between true and predicted labels.
    """
    y_pred = torch.round(y_pred)

    return torch.sum(y_pred == y_true) / len(y_true)

def get_diagonal_features(n_nodes):
    """ Get a sparse diagonal feature matrix.

    Parameters:
    -----------
    n_nodes : int
        The number of nodes.

    Returns:
    --------
    torch.sparse.FloatTensor
        A sparse diagonal feature matrix.

    Notes:
    ------
    - This function generates a sparse diagonal feature matrix with ones on the diagonal.
    """
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
    """ Initialize an output folder for experiment results.

    Parameters:
    -----------
    folder_name : str
        The name of the folder to create.

    Returns:
    --------
    None

    Notes:
    ------
    - This function removes an existing folder if it exists and then creates a new one.
    - It also initializes a training history logfile within the folder.
    """
    # Remove existing folder
    if os.path.exists(folder_name):
        shutil.rmtree(folder_name)
    
    # Re-make the experiment folder
    os.makedirs(folder_name)
    
    # Initialize training history logfile
    with open(folder_name + "/training_history.csv", "a") as f:
        f.write(f"Epoch,Train_loss,Train_acc")