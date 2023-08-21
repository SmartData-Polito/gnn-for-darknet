import torch
import torch
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
from ...utils.utils import (compute_accuracy, 
                            initalize_output_folder, 
                            get_diagonal_features)
from ...preprocessing.preprocessing import get_self_supervised_edges
from .modules._gcngru import GcnGruModule

class GCN_GRU():
    def __init__(self, n_nodes=1000, history=10, ns=1, input_size=None, 
                 gcn_layers=2, gcn_units=1024, gcn_output_size=512, 
                 embedding_size=128, predictor_units=64, dropout=.01, lr=1e-3, 
                 early_stop=None, best_train_acc=False, cuda=False):  
        # General parameters initializations
        self.n, self.ns, self.cuda = n_nodes, ns, cuda
        self.early_stop = early_stop
        self.best_train_acc = best_train_acc

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
        #training_batch contains (adjacency matrices and features)
        # [history][single training snapshot]
        
        X, features, positive_pairs = training_batch
        positive_pairs = positive_pairs[:, positive_pairs[0]<positive_pairs[1]]
        edges = torch.cat([positive_pairs, negative_pairs], dim=1)
                            
        #start_day = n_days - self.hist
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
        

    def fit(self, X, epochs, features=None, save=False, dir_name=None):
        # Initialize output folder
        if save:
            initalize_output_folder(dir_name)
        
        # X is the sparse adjacency matrix
        if self.cuda: X = [x.cuda() for x in X]
        
        # Preliminaries
        best_train_acc, best_model = 0.0, None
        training_days = len(X)-self.history
        
        # Set feature matrix to identity 
        if type(features)==type(None):
            features = [get_diagonal_features(self.n) for i in range(len(X))]
        
        no_improvement = 0
        for epoch in range(epochs):
            # General init at the beginning of each epoch
            tot_acc, tot_loss = .0, .0
            self.model.train()

            pbar = tqdm(range(training_days), desc=f'Epoch {epoch}')

            for it in pbar:
                # Look back "history" days and predict the next
                X_to_predict = X[self.history+it]
                X_neg, index = get_self_supervised_edges(
                                            X_to_predict, 
                                            cuda=self.cuda, 
                                            ns=self.ns
                )
                # Define training batch. Note +1 because in ":x" x is escluded
                training_batch = (
                    X[it:self.history+it+1], 
                    features[it:self.history+it+1], 
                    index
                )

                # Run single training step
                train_loss, train_acc = self._train_single_step_edges(
                                            training_batch, 
                                            X_neg
                )
                # Compute training metrics
                tot_loss += train_loss.detach().item() / (len(X))
                tot_acc  += train_acc.detach().item() / (len(X))     

                pbar.set_postfix({'train_loss': tot_loss, 
                                  'train_acc':tot_acc})

            if save:
                # Update training history
                with open(dir_name + "/training_history.csv", "a") as f:
                    f.write(f"\n{epoch},{tot_loss:.4},{tot_acc:.4}")
                    
            # Take best performing model
            if train_acc > best_train_acc: 
                best_train_acc = train_acc
                best_model = deepcopy(self.model)
                no_improvement = 0
            else:
                no_improvement += 1

            if type(self.early_stop)!=type(None):
                if no_improvement>=self.early_stop:
                    print('Early stop condition met')
                    break

        if save:
            if self.best_train_acc:
                # Save best model and best results for validation results
                torch.save(best_model.state_dict(), dir_name + "/best_model")
                self.model.load_state_dict(torch.load(dir_name + "/best_model"))
            else:
                # Save best model and best results for validation results
                torch.save(self.model.state_dict(), dir_name + "/best_model")
        else:
            if self.best_train_acc:
                self.model = best_model