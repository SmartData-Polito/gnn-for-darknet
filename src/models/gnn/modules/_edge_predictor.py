from torch import nn

class EdgePredictor(nn.Module):
    def __init__(self, input_size, units):
        super(EdgePredictor, self).__init__()
        
        # Define the edge predictor as a sequential module
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * input_size, units),
            nn.ReLU(),
            nn.Linear(units, 2),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        # Pass the input through the edge predictor
        return self.edge_predictor(x)