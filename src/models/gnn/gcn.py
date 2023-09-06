import torch
import torch
import torch.nn.functional as F
from ...utils.utils import (compute_accuracy, 
                            initalize_output_folder, 
                            get_diagonal_features)
from ...preprocessing.preprocessing import get_self_supervised_edges
from .modules._single_gcn import SingleGCN

class GCN():
    def __init__(self, n_nodes=1000, ns=1, input_size=None, 
                 gcn_layers=2, gcn_units=1024, gcn_output_size=256, 
                 embedding_size=128, predictor_units=64, dropout=.01, 
                 lr=1e-3, epochs=3, cuda=False):  
        """ This class implements the GCN (Graph Convolutional Network) model 
        with the specified hyperparameters.

        Parameters:
        -----------
        n_nodes : int, optional (default=1000)
            Number of nodes in the graph.

        ns : int, optional (default=1)
            Neighborhood size for self-supervised learning.

        input_size : int or None, optional (default=None)
            Input size for the model. If None, it is automatically determined.

        gcn_layers : int, optional (default=2)
            Number of GCN layers in the model.

        gcn_units : int, optional (default=1024)
            Number of units in each GCN layer.

        gcn_output_size : int, optional (default=256)
            Output size of the final GCN layer.

        embedding_size : int, optional (default=128)
            Size of the node embeddings.

        predictor_units : int, optional (default=64)
            Number of units in the predictor layer.

        dropout : float, optional (default=0.01)
            Dropout probability for regularization.

        lr : float, optional (default=1e-3)
            Learning rate for model training.

        epochs : int, optional (default=3)
            Number of training epochs.

        cuda : bool, optional (default=False)
            Whether to use CUDA for GPU acceleration.

        """
        # General parameters initializations
        self.n, self.ns, self.cuda, self.epochs = n_nodes, ns, cuda, epochs
        
        # Initialize the GCN-GRU model
        self.model = SingleGCN(
            n_nodes = n_nodes, 
            input_size = input_size, 
            gcn_layers = gcn_layers, 
            gcn_units = gcn_units, 
            gcn_output_size = gcn_output_size,
            embedding_size = embedding_size, 
            predictor_units = predictor_units, 
            dropout = dropout, 
            cuda = cuda
        )

        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=lr, # Learning rate 
            weight_decay=5e-3
        )

        # Manage CUDA if available and specified
        if self.cuda: self.model = self.model.cuda()
    
    def _train_single_step_edges(self, training_batch, negative_pairs):        
        """ Train the model for a single step using edge-level training data.

        Parameters:
        -----------
        training_batch : tuple
            A tuple containing training data, features, and positive pairs.

        negative_pairs : torch.Tensor
            Tensor containing negative pairs for training.

        Returns:
        --------
        train_loss : torch.Tensor
            The computed training loss for the current step.

        accuracy : float
            The computed 'local' accuracy for the current step.

        Notes:
        ------
        This method trains the model for a single step using edge-level training data.
        It computes predictions, retrieves self-supervised true labels, computes the loss,
        and backpropagates to update the model's parameters. It also computes the 'local'
        accuracy based on the predictions.

        """
        X, features, positive_pairs = training_batch
        positive_pairs = positive_pairs[:, positive_pairs[0]<positive_pairs[1]]
        edges = torch.cat([positive_pairs, negative_pairs], dim=1)
                            
        self.optimizer.zero_grad()   
        
        # Get predictions
        y_pred = self.model(X, edges, features=features)

        # Retrieve self-supervised true labels
        y_true = torch.cat([torch.ones([negative_pairs.shape[1] // self.ns,]), 
                            torch.zeros([negative_pairs.shape[1],])]).long()

        if self.cuda:
            y_true = y_true.cuda()
        
        # Compute loss 
        train_loss = F.nll_loss(y_pred, y_true)
        train_loss.backward()
        self.optimizer.step()

        # Compute 'local' accuracy
        _y_pred = torch.exp(y_pred[:,1].view(-1,))
        accuracy = compute_accuracy(y_true, _y_pred)

        return train_loss, accuracy
    
    def get_embeddings(self, X, features=None):
        """ Get embeddings for the given data.

        Parameters:
        -----------
        X : torch.Tensor
            Input data for which embeddings are to be obtained.

        features : torch.Tensor or None, optional (default=None)
            Feature matrix. If None, an identity matrix is used as features.

        Returns:
        --------
        embeddings : numpy.ndarray
            Embeddings obtained for the input data.

        Notes:
        ------
        This method computes embeddings for the input data using the model. It also
        allows for the specification of a feature matrix. If no feature matrix is provided,
        an identity matrix is used as features. The resulting embeddings are converted to
        a NumPy array for convenience.

        """
        # Set feature matrix to identity 
        if type(features)==type(None):
            features = get_diagonal_features(self.n)

        embeddings = self.model(X, features=features)

        return embeddings.cpu().detach().numpy()
        

    def fit(self, X, features=None, save=False, dir_name=None, pbar=None, day=None):
        """ Train the model for self-supervised learning.

        Parameters:
        -----------
        X : torch.Tensor
            The sparse adjacency matrix used for training.

        features : torch.Tensor or None, optional (default=None)
            Feature matrix. If None, an identity matrix is used as features.

        save : bool, optional (default=False)
            Whether to save the trained model and results.

        dir_name : str or None, optional (default=None)
            Directory name for saving the model and results.

        pbar : tqdm.tqdm or None, optional (default=None)
            Progress bar for tracking training progress.

        day : int or None, optional (default=None)
            The current day in the training process.

        Notes:
        ------
        This method trains the model for self-supervised learning using the provided
        adjacency matrix. It iterates through epochs and predicts future connections
        in the graph. If 'save' is True, it saves the trained model and results in the
        specified directory. 'pbar' is used to track training progress, and 'day' is
        used to indicate the current day in the training process.

        """
        # Initialize output folder
        if save:
            initalize_output_folder(dir_name)
        
        # X is the sparse adjacency matrix
        if self.cuda: X = X.cuda()
        
        # Set feature matrix to identity 
        if type(features)==type(None):
            features = get_diagonal_features(self.n)
        
        for epoch in range(self.epochs):
            # General init at the beginning of each epoch
            self.model.train()

            # Look back "history" days and predict the next
            X_to_predict = X
            X_neg, index = get_self_supervised_edges(
                                        X_to_predict, 
                                        cuda=self.cuda, 
                                        ns=self.ns
            )
            # Define training batch. Note +1 because in ":x" x is escluded
            training_batch = (
                X, 
                features, 
                index
            )

            # Run single training step
            self._train_single_step_edges(training_batch, X_neg)     
            
            # Update progress bar
            if type(pbar) != type(None):
                pbar.set_description(f"Day {day} - Epoch {epoch}")
                pbar.update(1)

        if save:
            # Save best model and best results for validation results
            torch.save(self.model.state_dict(), dir_name)