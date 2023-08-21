from torch import nn
import torch
import math
import torch.nn.functional as F
from tqdm import tqdm

class GraphConvolution(nn.Module):
    """ Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        
        # Define the input and output features for the layer
        self.in_features = in_features
        self.out_features = out_features

        # Create learnable weights and biases for the layer
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize the layer's parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Reset the weight and bias parameters with initial random values
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features, adj):
        # Compute the support (i.e., nodes features) for the layer
        support = torch.matmul(input_features, self.weight)

        # Perform graph convolution: 
        # multiply the adjacency matrix with the support
        output = torch.sparse.mm(adj, support) 

        # If bias is set, add the bias term to the output
        if self.bias is not None:
            return output + self.bias
        else:
            return output
        
        

class GraphConvolutionalNetwork(nn.Module):
    #def __init__(self, nfeat, nout, nhid, dropout, n_layers):
    def __init__(self, input_size, output_size, units, dropout, n_layers):
        super(GraphConvolutionalNetwork, self).__init__()
        
        # Number of layers in the GCN
        self.nlayers = n_layers

        # Define input and output layers sizes
        layer_input_sizes = [input_size] + [units] * (self.nlayers - 1)
        layer_output_sizes = [units] * (self.nlayers - 1) + [output_size]
        
        # Initialize the layers
        self.layers = nn.ModuleList([
            GraphConvolution(
                layer_input_sizes[i], 
                layer_output_sizes[i]) 
            for i in range(self.nlayers)
        ])

        # Get the dropout probability
        self.dropout = dropout

    def forward(self, features, adjacency):
        x = features # Get the node features
        for i in range(self.nlayers):
            # Pass the input through each GCN layer
            x = self.layers[i](x, adjacency)
            
            # Apply ReLU activation for all layers except the last one
            if i < self.nlayers - 1:
                x = F.relu(x) 
            
            # Apply dropout to the output of each layer during training
            x = F.dropout(x, self.dropout, training=self.training)

        return x