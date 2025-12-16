import torch
import torchfd
import torch.nn as nn
import torch.nn.functional as F


class DotProductClassifier(torchfd.mutual_information.MINE):
    """
    Ultra-simple classifier - just dot product
    Parameters: ~0 (only projector weights)
    Complexity: O(d)
    """
    def __init__(self, marginalizer=None):
        super().__init__(marginalizer=marginalizer)
    
    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x, y):
        return torch.sum(x * y, dim=-1)


class BilinearClassifier(torchfd.mutual_information.MINE):
    """
    Medium complexity - bilinear interaction
    Parameters: O(d²)
    Complexity: medium
    """
    def __init__(self, X_dim, Y_dim, marginalizer=None):
        super().__init__(marginalizer=marginalizer)
        self.bilinear = nn.Bilinear(X_dim, Y_dim, 1)
    
    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x, y):
        return self.bilinear(x, y)
    

class VeryDenseT(torchfd.mutual_information.MINE):
    """
    Ultra-complex classifier - deep MLP
    Parameters: O(4×d²) 
    Complexity: very high
    """
    def __init__(self, X_dim, Y_dim, inner_dim=512, n_layers=4, dropout=0.1, marginalizer=None):
        super().__init__(marginalizer=marginalizer)
        
        layers = []
        input_dim = X_dim + Y_dim
        
        # Multiple hidden layers
        for i in range(n_layers):
            layers.extend([
                nn.Linear(input_dim, inner_dim),
                nn.BatchNorm1d(inner_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            input_dim = inner_dim
        
        # Final layer
        layers.append(nn.Linear(inner_dim, 1))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain('leaky_relu'))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x, y):
        return self.net(torch.cat([x, y], dim=-1))


# Factory function to get all available architectures
def get_architecture_factory(embedding_dim, config):
    """
    Returns dictionary with all available architectures
    """
    from . import modules  # Import existing modules
    
    return {
        # New architectures
        "DotProduct": lambda: DotProductClassifier(
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
        "Bilinear": lambda: BilinearClassifier(
            embedding_dim, embedding_dim,
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
        "VeryDenseT": lambda: VeryDenseT(
            embedding_dim, embedding_dim,
            inner_dim=512, n_layers=4,
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
        
        # Existing architectures from modules.py
        "AdditiveGaussainT": lambda: modules.AdditiveGaussainT(
            p=0.99,
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
        "SeparableT": lambda: modules.SeparableT(
            embedding_dim, embedding_dim,
            inner_dim=config.get("discriminator_network_inner_dim", 128),
            output_dim=config.get("discriminator_network_output_dim", 64),
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
        "DenseT": lambda: modules.DenseT(
            embedding_dim, embedding_dim,
            inner_dim=config.get("discriminator_network_inner_dim", 256),
            marginalizer=getattr(torchfd.mutual_information, config["marginalizer"])()
        ),
    }
