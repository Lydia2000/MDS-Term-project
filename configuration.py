import torch
from torch import nn
from lstm import LSTMModel

class Configuration():
    def __init__(self, preprocessing_method, output_dir="output"):
        
        # Configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyper parameter setting
        self.batch_size = 64
        self.learning_rate = 0.001
        self.n_components = 10

        if preprocessing_method == 'pca':
            self.n_features = self.n_components  # number of feature
        elif preprocessing_method == 'en':
            pass

        self.hidden_dim = 10
        self.output_dim = 1
        self.n_layers = 4 
        self.dropout = 0.05
        self.epochs = 100 # 100
        self.window_size = 20

        # Initial model
        self.model_name = 'lstm' # other model can be added
        self.model = LSTMModel(
            input_dim = self.n_features,
            n_layers = self.n_layers,
            hidden_dim = self.hidden_dim,
            output_dim = self.output_dim,
            dropout = self.dropout,
            device= self.device,
        ).to(self.device)

        # Optimizer (Adam, AdamW, SGD)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        