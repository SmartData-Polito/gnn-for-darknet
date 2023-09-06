from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import joblib

class KnnClassifier():
    def __init__(self, n_neighbors=7, model_path=None, metric='cosine', 
                 _load_model=False):
        """ This class defines the KNN Classifier with specified hyperparameters 
        and creates a StandardScaler for data standardization. If `_load_model` 
        is True, it loads a pre-trained model and scaler from the specified path.

        Parameters:
        -----------
        n_neighbors : int, optional (default=7)
            Number of neighbors to consider when making predictions.

        model_path : str or None, optional (default=None)
            Path to save or load the trained model and scaler. If None, no 
            saving or loading is performed.

        metric : str, optional (default='cosine')
            The distance metric used for nearest neighbor computation.

        _load_model : bool, optional (default=False)
            Whether to load a pre-trained model and scaler from the specified 
            path.

        """
        self.model_path=model_path
        self.scaler = StandardScaler()
        self.neighbors, self.y_train = None, None
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, 
                                          metric=metric, n_jobs=-1)
                    
        if _load_model:
            self.scaler = joblib.load(f'{self.model_path}_knn_scaler.save')
            self.X, self.y = joblib.load(f'{self.model_path}_knn.save')
    
    def _scale_data(self, X_train, X_val=None):
        """ Scale the input data using the fitted scaler.

        Parameters:
        -----------
        X_train : array-like
            The training data to be scaled.

        X_val : array-like or None, optional (default=None)
            The validation data to be scaled. If None, no scaling is applied to 
            validation data.

        Returns:
        --------
        X_train_scaled : array-like
            Scaled training data.

        X_val_scaled : array-like or None
            Scaled validation data if provided; None if X_val is None.

        Notes:
        ------
        This method fits the scaler on the training data and then scales both 
        the training and validation data using the same scaler. If no validation 
        data is provided, only the training data is scaled.

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
        """ Train a classifier on the given data and labels.

        Parameters:
        -----------
        X : array-like
            The training data.

        y : array-like
            The corresponding labels.

        scale_data : bool, optional (default=True)
            Whether to scale the input data before training the classifier.

        save : bool, optional (default=True)
            Whether to save the trained model and scaler to files.

        Notes:
        ------
        This method trains a classifier on the input data and labels. If 
        `scale_data` is True, it performs data standardization. If `save` is 
        True, it saves the trained model and scaler to files with appropriate 
        names.

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
        """ Perform majority voting to make predictions based on neighbor labels.

        Parameters:
        -----------
        neigh_idxs : array-like
            Indices of neighbors in the training data.

        Returns:
        --------
        predictions : array-like
            Predicted labels based on majority voting.

        Notes:
        ------
        This method takes the indices of neighbors in the training data and 
        performs majority voting to make predictions. It calculates the most 
        frequent label among neighbors and returns it as the prediction.

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
        """Make predictions using the K-Nearest Neighbors (KNN) Classifier.

        Parameters:
        -----------
        X : array-like
            The samples for which predictions are to be made.

        scale_data : bool, optional (default=True)
            Whether to scale the input data before making predictions.

        loo : bool, optional (default=False)
            Whether to perform Leave-One-Out validation. If True, 'X' should be 
            a numpy array with indices for Leave-One-Out validation.

        Returns:
        --------
        y_pred : array-like
            Predicted labels for the input samples.

        Notes:
        ------
        This method makes predictions using the KNN Classifier. If 'loo' is 
        True, it performs Leave-One-Out validation using the provided indices 
        in 'X'. Otherwise, it performs classic fit-predict. If 'scale_data' is 
        True, the input data is scaled before making predictions.

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
        """ Predict class probabilities for the given samples.

        Parameters:
        -----------
        X : array-like
            The samples for which class probabilities are to be predicted.

        Returns:
        --------
        probas : array-like
            Class probabilities for each sample.

        Notes:
        ------
        This method predicts class probabilities for the given samples. It 
        calculates the probability of each sample belonging to the class that 
        is most frequent among its nearest neighbors. The class probabilities 
        are normalized by the number of nearest neighbors (N).

        """
        y_neigh = self.y[self.model.kneighbors()[1][X]]
        y_true  = self.y[X]
        N = self.model.n_neighbors
        pairs = zip(y_true, y_neigh)
        probas = [np.where(b==a)[0].shape[0]/N for a,b in pairs]
        probas = np.asarray(probas)
        
        return probas