import torch
import torch
import torch.nn.functional as F
from ...utils.utils import (compute_accuracy, 
                            initalize_output_folder, 
                            get_diagonal_features)
from ...preprocessing.preprocessing import get_self_supervised_edges
from .modules._gcngru import GcnGruModule

class IncrementalGcnGru():
    def __init__(self, n_nodes=1000, history=10, ns=1, input_size=None, 
                 gcn_layers=2, gcn_units=1024, gcn_output_size=512,
                 embedding_size=128, predictor_units=64, dropout=.01, lr=1e-3, 
                 epochs=3, cuda=False):  
        # General parameters initializations
        self.n, self.ns, self.cuda, self.epochs = n_nodes, ns, cuda, epochs

        # Number of days to use as history
        self.history = history

        # Initialize the GCN-GRU model
        self.model = GcnGruModule(
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
        # Set feature matrix to identity 
        if type(features)==type(None):
            features = [get_diagonal_features(self.n) for i in range(len(X))]

        embeddings = self.model(X, features=features)

        return embeddings.cpu().detach().numpy()
        

    def fit(self, X, features=None, save=False, dir_name=None, pbar=None, day=None):
        # Initialize output folder
        if save:
            initalize_output_folder(dir_name)
        
        # X is the sparse adjacency matrix
        if self.cuda: X = [x.cuda() for x in X]
        
        # Set feature matrix to identity 
        if type(features)==type(None):
            features = [get_diagonal_features(self.n) for i in range(len(X))]
        
        for epoch in range(self.epochs):
            # General init at the beginning of each epoch
            self.model.train()

            # Look back "history" days and predict the next
            X_to_predict = X[self.history]
            X_neg, index = get_self_supervised_edges(
                                        X_to_predict, 
                                        cuda=self.cuda, 
                                        ns=self.ns
            )
            # Define training batch. Note +1 because in ":x" x is escluded
            training_batch = (
                X[:self.history+1], 
                features[:self.history+1], 
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