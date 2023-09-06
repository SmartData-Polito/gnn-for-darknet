# `src.models.nlp.iWord2Vec`

# `src.models.classification.KnnClassifier`

```python
class iWord2Vec(c=5, e=64, epochs=1, source=None, destination=None, seed=15)
```

This class implements a iWord2Vec model.

### Parameters

- **c** : _int, optional (default=5)_

    The size of the context window.

- **e** : _int, optional (default=64)_

    The size of the word embeddings.

- **epochs** : _int, optional (default=1)_

    The number of training epochs.

- **source** : _str or None, optional (default=None)_

    The source file to load a pre-trained model from.

- **destination** : _str or None, optional (default=None)_

    The destination file to save the trained model.

- **seed** : _int, optional (default=15)_

    The random seed for reproducibility.

### Methods
```python
train(corpus, save=False)
```
Train the iWord2Vec model on the given corpus.

#### Parameters
- **corpus** : _list of list of str_
    A list of sentences where each sentence is a list of words.

- **save** : _bool, optional (default=False)_
    Whether to save the trained model.

___
```python
load_model():
```
Load a pre-trained iWord2Vec model from a file.

___
```python
get_embeddings(ips=None, emb_path=None)
```
Get word embeddings for specific words or all words.

#### Parameters
- **ips** : _list of str or None, optional (default=None)_
    A list of words to retrieve embeddings for. If None, retrieves embeddings for all words.

- **emb_path** : _str or None, optional (default=None)_
    The file path to save the embeddings as a CSV file.

#### Returns
- **embeddings** : _pd.DataFrame_

A DataFrame containing word embeddings.

___
```python
update(corpus, save=False)
```
Update the iWord2Vec model with additional training on a new corpus.

#### Parameters
- **corpus** : _list of list of str_
    A list of sentences where each sentence is a list of words.

- **save** : _bool, optional (default=False)_
    Whether to save the updated model.

___
```python
del_embeddings(to_drop, mname=None)
```
Delete word embeddings for specific words.

#### Parameters
- **to_drop** : _list of str_
    A list of words to delete from the embeddings.

- **mname** : _str or None, optional (default=None)_
    The destination file to save the model after removing embeddings.