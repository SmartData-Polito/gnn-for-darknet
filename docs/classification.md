# `src.models.classification.KnnClassifier`

```python
class KnnClassifier(n_neighbors=7, model_path=None, metric='cosine', _load_model=False)
```

This class defines the KNN Classifier with specified hyperparameters 
        and creates a StandardScaler for data standardization. If `_load_model` 
        is True, it loads a pre-trained model and scaler from the specified path.

**Parameters**

- **n_neighbors** : int, optional (default=7)
    
    Number of neighbors to consider when making predictions.

- **model_path** : str or None, optional (default=None)
    
    Path to save or load the trained model and scaler. If None, no 
    saving or loading is performed.

- **metric** : str, optional (default='cosine')
    
    The distance metric used for nearest neighbor computation.

- **_load_model** : bool, optional (default=False)
    
    Whether to load a pre-trained model and scaler from the specified 
    path.