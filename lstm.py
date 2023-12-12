import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, n_layers, hidden_dim, output_dim, dropout,
                 device ,bidirectional=False):
        super(LSTMModel, self).__init__()

        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        
        # Dimensions
        self.input_dim = input_dim # number of feature
        self.hidden_dim = hidden_dim

        # Number of hidden layers
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size = self.input_dim, 
            hidden_size = self.hidden_dim,
            num_layers = self.n_layers,
            dropout = self.dropout,
            bidirectional = bidirectional
            )
        
        # Initialize LSTM weights using Xavier/Glorot initialization
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        

    def forward(self, x):

        self.lstm.flatten_parameters()

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.n_layers, x.size(1), self.hidden_dim).to(self.device)

        # Initialize cell state
        c0 = torch.zeros(self.n_layers, x.size(1), self.hidden_dim).to(self.device)

        output, (hidden_state, cell_state) = self.lstm(x, (h0, c0))
        output = self.fc(output[-1, :, :])
        
        # output.shape is (64, 1)
        return output
