from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib

class KnnClassifier():
    """_summary_

    Parameters
    ----------
    n_neighbors : int, optional
        _description_, by default 7
    model_path : _type_, optional
        _description_, by default None
    metric : str, optional
        _description_, by default 'cosine'
    _load_model : bool, optional
        _description_, by default False
    """
    def __init__(self, n_neighbors=7, model_path=None, metric='cosine', 
                 _load_model=False):
        self.model_path=model_path
        self.scaler = StandardScaler()
        self.neighbors, self.y_train = None, None
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                          metric=metric, n_jobs=-1)
                    
        if _load_model:
            self.scaler = joblib.load(f'{self.model_path}_knn_scaler.save')
            self.X, self.y = joblib.load(f'{self.model_path}_knn.save')
    
    def _scale_data(self, X_train, X_val=None):
        """_summary_

        Parameters
        ----------
        X_train : _type_
            _description_
        X_val : _type_, optional
            _description_, by default None

        Returns
        -------
        _type_
            _description_
        """
        # Fit the scaler on training data
        self.scaler.fit(X_train)
        # Scale training data
        X_train = self.scaler.transform(X_train)
        # If provided, scale validation data
        if type(X_val)!=type(None): 
            X_val = self.scaler.transform(X_val) 
            
        return X_train, X_val
    
    def fit(self, X, y, scale_data=True, save=False):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        y : _type_
            _description_
        scale_data : bool, optional
            _description_, by default True
        save : bool, optional
            _description_, by default False
        """
        self.X, self.y = X, y  
        # Data standardization
        if scale_data:
            X, _ = self._scale_data(X)
        # Save the best model according to the max val_accuracy
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self.model.fit(X, y)      
        # Train the classifier
        if save:
            joblib.dump([X, y], f'{self.model_path}_knn.save')
            joblib.dump(self.scaler, f'{self.model_path}_knn_scaler.save')

    def _majority_voting(self, neigh_idxs):
        """_summary_

        Parameters
        ----------
        neigh_idxs : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        neigh_labels = self.y[neigh_idxs]
        predictions = []
        for sample in neigh_labels:
            labs, freqs = np.unique(sample, return_counts=True)
            idx = np.argmax(freqs)
            try: 
                idx.shape[0]
                if type(labs) == str:
                    predictions.append('n.a.')
                else:
                    predictions.append(-1)
            except IndexError as e:
                predictions.append(labs[idx])

        return np.asarray(predictions)
    
    def predict(self, X, scale_data=True, loo=False):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_
        scale_data : bool, optional
            _description_, by default True
        loo : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_
        """
        if loo: # Leave-One-Out validation - X is a numpy array with indices
            neighbors = self.model.kneighbors()[1][X]
            y_pred = self._majority_voting(neighbors)
        else: # Classic fit-predict
            if scale_data:
                X = self.scaler.transform(X)
            y_pred = self.model.predict(X)
        
        return y_pred

    def predict_proba(self, X):
        """_summary_

        Parameters
        ----------
        X : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        y_neigh = self.y[self.model.kneighbors()[1][X]]
        y_true  = self.y[X]
        N = self.model.n_neighbors
        pairs = zip(y_true, y_neigh)
        probas = [np.where(b==a)[0].shape[0]/N for a,b in pairs]
        probas = np.asarray(probas)
        
        return probas