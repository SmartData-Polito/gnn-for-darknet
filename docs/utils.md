# `src.utils`

```python
_normalize(mx)
```
[source](../src/utils/utils.py)

This function row-normalizes a sparse matrix by dividing each row by its sum.
Avoids division by zero by adding a small epsilon (1e-15) to the denominator.

#### Parameters

- **mx** : _sp.csr_matrix_

    The input sparse matrix.

#### Returns

- _sp.csr_matrix_

    The row-normalized sparse matrix.

___

```python
_sparse_mx_to_torch_sparse_tensor(sparse_mx)
```
[source](../src/utils/utils.py)

This function converts a scipy sparse matrix into a corresponding torch sparse tensor.

#### Parameters

- **sparse_mx** : _sp.csr_matrix_

    The input scipy sparse matrix.

#### Returns

- _torch.sparse.FloatTensor_

    The torch sparse tensor equivalent of the input matrix.
___

```python
get_set_diff(A,B)
```
[source](../src/utils/utils.py)

Compute the set difference between two arrays A and B.

#### Parameters

- **A** : _np.ndarray_

    The first input array.
- **B** : _np.ndarray_

    The second input array.

#### Returns

- _np.ndarray_

    An array containing the set difference between A and B.

___

```python
compute_accuracy(y_true, y_pred)
```
[source](../src/utils/utils.py)

Compute accuracy between true and predicted labels.

#### Parameters

- **y_true** : _torch.Tensor_

    The true labels.
- **y_pred** : _torch.Tensor_

    The predicted labels.

#### Returns

- _torch.Tensor_

    The accuracy score.

___

```python
get_diagonal_features(n_nodes)
```
[source](../src/utils/utils.py)

Get a sparse diagonal feature matrix.

#### Parameters

- **n_nodes** : _int_

    The number of nodes.

#### Returns

- _torch.sparse.FloatTensor_

    A sparse diagonal feature matrix.
___

```python
initalize_output_folder(folder_name)
```
[source](../src/utils/utils.py)

Initialize an output folder for experiment results.

#### Parameters

- **folder_name** : _str_

    The name of the folder to create.