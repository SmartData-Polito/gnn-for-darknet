# `src.models.classification.KnnClassifier`

```python
class KnnClassifier(n_neighbors=7, model_path=None, metric='cosine', _load_model=False)
```

This class defines the KNN Classifier with specified hyperparameters 
        and creates a StandardScaler for data standardization. If `_load_model` 
        is True, it loads a pre-trained model and scaler from the specified path.

### Parameters

- **n_neighbors** : _int, optional (default=7)_
    
    Number of neighbors to consider when making predictions.

- **model_path** : _str or None, optional (default=None)_
    
    Path to save or load the trained model and scaler. If None, no 
    saving or loading is performed.

- **metric** : _str, optional (default='cosine')_
    
    The distance metric used for nearest neighbor computation.

- **_load_model** : _bool, optional (default=False)_
    
    Whether to load a pre-trained model and scaler from the specified 
    path.

### Methods
```python
_scale_data(X_train, X_val=None)
```
This method fits the scaler on the training data and then scales both 
        the training and validation data using the same scaler. If no validation 
        data is provided, only the training data is scaled.

#### Parameters
- **X_train** : _array-like_

The training data to be scaled.

- **X_val** : _array-like or None, optional (default=None)_

The validation data to be scaled. If None, no scaling is applied to 
validation data.

#### Returns
- **X_train_scaled** : _array-like_

Scaled training data.

- **X_val_scaled** : _array-like or None_

Scaled validation data if provided; None if X_val is None.


```python
fit(X, y, scale_data=True, save=False):
```
This method trains a classifier on the input data and labels. If 
`scale_data` is True, it performs data standardization. If `save` is 
True, it saves the trained model and scaler to files with appropriate 
names.

#### Parameters
- **X** : _array-like_

The training data.

- **y** : _array-like_

The corresponding labels.

- **scale_data** : _bool, optional (default=True)_

Whether to scale the input data before training the classifier.

- **save** : _bool, optional (default=True)_

Whether to save the trained model and scaler to files.


```python
_majority_voting(neigh_idxs)
```
This method takes the indices of neighbors in the training data and 
performs majority voting to make predictions. It calculates the most 
frequent label among neighbors and returns it as the prediction.

#### Parameters
- **neigh_idxs** : _array-like_

Indices of neighbors in the training data.

#### Returns
- **predictions** : _array-like_

Predicted labels based on majority voting.


```python
predict(X, scale_data=True, loo=False)
```
This method makes predictions using the KNN Classifier. If 'loo' is 
True, it performs Leave-One-Out validation using the provided indices 
in 'X'. Otherwise, it performs classic fit-predict. If 'scale_data' is 
True, the input data is scaled before making predictions.

#### Parameters
- **X** : _array-like_

The samples for which predictions are to be made.

- **scale_data** : _bool, optional (default=True)_

Whether to scale the input data before making predictions.

- **loo** : _bool, optional (default=False)_

Whether to perform Leave-One-Out validation. If True, 'X' should be 
a numpy array with indices for Leave-One-Out validation.

#### Returns
- **y_pred** : _array-like_

Predicted labels for the input samples.


```python
predict_proba(X)
```
This method predicts class probabilities for the given samples. It 
calculates the probability of each sample belonging to the class that 
is most frequent among its nearest neighbors. The class probabilities 
are normalized by the number of nearest neighbors (N).

#### Parameters
- **X** : _array-like_

The samples for which class probabilities are to be predicted.

#### Returns
- **probas** : _array-like_

Class probabilities for each sample.