# `src.models.gnn.GCN`

```python
class GCN(n_nodes=1000, ns=1, input_size=None, gcn_layers=2, gcn_units=1024,  
          gcn_output_size=256, embedding_size=128, predictor_units=64, 
          dropout=.01, lr=1e-3, epochs=3, cuda=False)
```

This class implements the GCN (Graph Convolutional Network) model 
with the specified hyperparameters.

### Parameters

- **n_nodes** : _int, optional (default=1000)_
    
    Number of nodes in the graph.

- **ns** : _int, optional (default=1)_

    Neighborhood size for self-supervised learning.

- **input_size** : _int or None, optional (default=None)_

    Input size for the model. If None, it is automatically determined.

- **gcn_layers** : _int, optional (default=2)_

    Number of GCN layers in the model.

- **gcn_units** : _int, optional (default=1024)_

    Number of units in each GCN layer.

- **gcn_output_size** : _int, optional (default=256)_

    Output size of the final GCN layer.

- **embedding_size** : _int, optional (default=128)_

    Size of the node embeddings.

- **predictor_units** : _int, optional (default=64)_

    Number of units in the predictor layer.

- **dropout** : _float, optional (default=0.01)_

    Dropout probability for regularization.

- **lr** : _float, optional (default=1e-3)_

    Learning rate for model training.

- **epochs** : _int, optional (default=3)_

    Number of training epochs.

- **cuda** : _bool, optional (default=False)_

    Whether to use CUDA for GPU acceleration.

### Methods

```python
_train_single_step_edges(training_batch, negative_pairs)
```

This method trains the model for a single step using edge-level training data.
It computes predictions, retrieves self-supervised true labels, computes the loss,
and backpropagates to update the model's parameters. It also computes the 'local'
accuracy based on the predictions.

#### Parameters
- **training_batch** : _tuple_

    A tuple containing training data, features, and positive pairs.

- **negative_pairs** : _torch.Tensor_

    Tensor containing negative pairs for training.

#### Returns
- **train_loss** : _torch.Tensor_

    The computed training loss for the current step.

- **accuracy** : _float_

    The computed 'local' accuracy for the current step.

___

```python
get_embeddings(X, features=None)
```

This method computes embeddings for the input data using the model. It also
allows for the specification of a feature matrix. If no feature matrix is provided,
an identity matrix is used as features. The resulting embeddings are converted to
a NumPy array for convenience.

#### Parameters

- **X** : _torch.Tensor_

    Input data for which embeddings are to be obtained.

- **features** : _torch.Tensor or None, optional (default=None)_

    Feature matrix. If None, an identity matrix is used as features.

#### Returns

- **embeddings** : _numpy.ndarray_

    Embeddings obtained for the input data.

___
```python
fit(X, features=None, save=False, dir_name=None, pbar=None, day=None)
```
This method trains the model for self-supervised learning using the provided
adjacency matrix. It iterates through epochs and predicts future connections
in the graph. If 'save' is True, it saves the trained model and results in the
specified directory. 'pbar' is used to track training progress, and 'day' is
used to indicate the current day in the training process.
        
#### Parameters

- **X** : _torch.Tensor_

    The sparse adjacency matrix used for training.

- **features** : _torch.Tensor or None, optional (default=None)_

    Feature matrix. If None, an identity matrix is used as features.

- **save** : _bool, optional (default=False)_

    Whether to save the trained model and results.

- **dir_name** : _str or None, optional (default=None)_

    Directory name for saving the model and results.

- **pbar** : _tqdm.tqdm or None, optional (default=None)_

    Progress bar for tracking training progress.

- **day** : _int or None, optional (default=None)_

    The current day in the training process.


# `src.models.gnn.GCN_GRU`

# `src.models.gnn.IncrementalGcnGru`