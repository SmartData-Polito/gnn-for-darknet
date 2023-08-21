from torch import nn

class GatedRecurrentUnit(nn.Module):
    def __init__(self, input_size, units, n_layers):
        super(GatedRecurrentUnit, self).__init__()
        
        self.gru = nn.GRU(
            input_size = input_size,  
            hidden_size = units, 
            num_layers = n_layers
        )

    def forward(self, x):
        return self.gru(x)
    
    def flatten_parameters(self):
        self.gru.flatten_parameters()