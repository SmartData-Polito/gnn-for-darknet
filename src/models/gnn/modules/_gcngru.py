import torch
from ._edge_predictor import EdgePredictor
from ._gcn import GraphConvolutionalNetwork
from ._gru import GatedRecurrentUnit

class GcnGruModule(torch.nn.Module):    
    def __init__(self, n_nodes=1000, input_size=None, gcn_layers=2, 
                 gcn_units=512, gcn_output_size=256, embedding_size=128, 
                 predictor_units=64, dropout=.15, cuda=False):
        super(GcnGruModule, self).__init__()
        self.n_nodes = n_nodes
        self.gcn_units = gcn_units
        self.is_cuda = cuda
        self.gcn_output_size = gcn_output_size
        self.embedding_size = embedding_size

        if type(input_size) == type(None):
            input_size = n_nodes

        # 1st module - GraphConvolutionalNetwork
        self.gcn = GraphConvolutionalNetwork(
            input_size = input_size, 
            output_size = gcn_output_size, 
            units=gcn_units, 
            dropout=dropout, 
            n_layers=gcn_layers
        )
        
        # 2nd module - Gated Recurrent Unit 
        self.gru = GatedRecurrentUnit(
            input_size = gcn_output_size, 
            units = embedding_size, 
            n_layers = 1
        )
        
        # 3rd module - Edge Predictor
        self.edge_predictor = EdgePredictor(
            input_size = embedding_size,
            units = predictor_units,
        )
        
        # Batch Normalization for embeddings
        if cuda:
            self.batch_norm = torch.nn.BatchNorm1d(self.embedding_size).cuda()
        else:
            self.batch_norm = torch.nn.BatchNorm1d(self.embedding_size)

        # Initialize weights    
        self._reset_weights(self.gru)  
        self._reset_weights(self.edge_predictor)
    
    
    def _reset_weights(self, model):
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.xavier_uniform(model.weight)
            model.bias.data.fill_(0.01)

    def forward(self, adjs, edges=None, features=None):           
        # Initialize the output of the GCN
        gcn_out = torch.zeros([len(adjs), self.n_nodes, self.gcn_output_size])
        if self.is_cuda:
            gcn_out = gcn_out.cuda()
        
        # Forward GCN on each daily snapshot
        for i in range(len(adjs)):
            if self.is_cuda:
                current_feature = features[i].float().cuda()
                adjacency_matrix = adjs[i].float().cuda()
            else:
                current_feature = features[i].float()
                adjacency_matrix = adjs[i].float()

            gcn_out[i] = self.gcn(current_feature, adjacency_matrix)                

        # Forward GCN outputs to GRU
        self.gru.flatten_parameters()
        _, embeddings = self.gru(gcn_out)
        embeddings = embeddings.squeeze()

        # Forward GRU output (i.e., embeddings) to batch normalization
        embeddings = self.batch_norm(embeddings)
        
        if type(edges)!=type(None):
            # Retrieve source node and destination node embeddings
            src_node_embedding = embeddings[edges[0,:].long(), :]
            dst_node_embedding = embeddings[edges[1,:].long(), :]
            
            # Predict edge likelihood
            pred = self.edge_predictor(torch.cat([
                src_node_embedding, 
                dst_node_embedding
            ], dim=1))

            return pred
        
        else:
            return embeddings