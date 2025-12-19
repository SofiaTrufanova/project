import torch
import math
from torchfd.mutual_information import MINE


import numpy
import scipy.linalg


class Sparse(MINE):
    def __init__(self, classifier: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        
        self.classifier = classifier  # энкодер
        self.head = head  # классификатор сравнения

    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)

        x_logits = self.classifier(x)
        y_logits = self.classifier(y)

        return self.head(x_logits, y_logits)


class Linear(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()

        self.dimension = dimension
        #self.linear = torch.nn.Linear(2 * self.dimension, 1)

    def forward(self, x_logits: torch.tensor, y_logits: torch.tensor) -> torch.tensor:
        #return self.linear(torch.cat([x_logits, y_logits], dim=-1))
        return torch.gather(x_logits, 1, torch.argmax(y_logits, dim=1, keepdim=True)) + torch.gather(y_logits, 1, torch.argmax(x_logits, dim=1, keepdim=True))


def covariance_matrix_to_mutual_information(matrix: torch.tensor, split_dim: int) -> torch.tensor:
    """
    Get mutual information from normalization matrix.
    
    Parameters
    ----------
    matrix : torch.tensor
        Normalization matrix.
    split_dim : int
        Split dimension.
    """
    
    return torch.slogdet(matrix[:split_dim,:split_dim])[1] + torch.slogdet(matrix[split_dim:,split_dim:])[1] - torch.slogdet(matrix)[1]


def mutual_information_to_correlation(mutual_information: torch.tensor) -> torch.tensor:
    """
    Calculate the absolute value of the correlation coefficient between two
    jointly Gaussian random variables given the value of mutual information.

    Parameters
    ----------
    mutual_information : torch.tensor
        Mutual information (lies in [0.0; +inf)).

    Returns
    -------
    correlation_coefficient : torch.tensor
        Corresponding correlation coefficient.
    """

    if (mutual_information < 0.0).any():
        raise ValueError("Mutual information must be non-negative")

    return torch.sqrt(1 - torch.exp(-2.0 * mutual_information))


def forward_coefficients_from_correlation(correlation_coefficient: torch.tensor) -> (torch.tensor, torch.tensor):
    """
    Get forward (colorizing) transformation matrix coefficients from correlation coefficient.
    
    Parameters
    ----------
    correlation_coefficient : torch.tensor
        Correlation coefficient (ranges from 0.0 to 1.0).
    """
    
    if (correlation_coefficient < 0.0).any() or (correlation_coefficient > 1.0).any():
        raise ValueError("Correlation coefficient must be in range [0.0, 1.0]")
    
    alpha = 0.5 * torch.sqrt(1.0 + correlation_coefficient)
    beta  = 0.5 * torch.sqrt(1.0 - correlation_coefficient)
    on_diagonal  = alpha + beta
    off_diagonal = alpha - beta
    
    return on_diagonal, off_diagonal


def inverse_coefficients_from_correlation(correlation_coefficient: torch.tensor) -> (torch.tensor, torch.tensor):
    """
    Get inverse (whitening) transformation matrix coefficients from correlation coefficient.
    
    Parameters
    ----------
    correlation_coefficient : torch.tensor
        Correlation coefficient (ranges from 0.0 to 1.0).
    """
    
    forward_on_diagonal, forward_off_diagonal = forward_coefficients_from_correlation(correlation_coefficient)
    
    denominator  =  torch.sqrt(1.0 - correlation_coefficient**2)
    on_diagonal  =  forward_on_diagonal  / denominator
    off_diagonal = -forward_off_diagonal / denominator
    
    return on_diagonal, off_diagonal


class GaussianMixture(MINE):
    def __init__(self, dimension: int, n_classes: int, sigma: float=1.0):
        """
        Create an instance of `GaussianMixture`

        Parameters
        ----------
        n_classes : int
            The number of classes.
        sigma : float, optional
            The scale parameter of the clusters.
        correlation_coefficient : float, optional
            The correlation parameter
        """

        if n_classes <= 0:
            raise ValueError("Expected `n_classes` to be strictly positive")
        
        if sigma <= 0.0:
            raise ValueError("Expected `sigma` to be strictly positive")
        
        super().__init__()

        self.n_classes = n_classes
        self.sigma = sigma
        self.dimension = dimension

        self.register_buffer("log_componentwise_mutual_information", torch.zeros((self.dimension,)) + math.log(3.0))# - math.log(self.dimension))
        self.shift = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

    @property
    def componentwise_mutual_information(self) -> torch.tensor:
        """ Componentwise mutual information. """
        return torch.exp(self.log_componentwise_mutual_information)

    @property
    def mutual_information(self) -> torch.tensor:
        """ Mutual information. """
        return torch.sum(self.componentwise_mutual_information)

    @property
    def correlation_coefficient(self) -> torch.tensor:
        """ Componentwise correlation coefficients. """
        return mutual_information_to_correlation(self.componentwise_mutual_information)


    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)
        
        dimension = self.dimension#x.shape[-1]

        means = torch.linspace(-1.0, 1.0, self.n_classes).repeat((dimension, 1)).to(x.device)
        centered_x = (x[...,None] - means[None,...]) / self.sigma
        centered_y = (y[...,None] - means[None,...]) / self.sigma

        correlation_coefficient = self.correlation_coefficient
        on_diagonal, off_diagonal = inverse_coefficients_from_correlation(correlation_coefficient)

        prior_x = centered_x * on_diagonal[None,:,None] + centered_y * off_diagonal[None,:,None]
        prior_y = centered_y * on_diagonal[None,:,None] + centered_x * off_diagonal[None,:,None]

        x_logits = torch.logsumexp(-0.5 * torch.sum(centered_x**2, dim=1), dim=-1)
        y_logits = torch.logsumexp(-0.5 * torch.sum(centered_y**2, dim=1), dim=-1)
        x_y_logits = torch.logsumexp(-0.5 * (torch.sum(prior_x**2, dim=1) + torch.sum(prior_y**2, dim=1)), dim=-1)

        logits = self.mutual_information + x_y_logits - x_logits - y_logits - self.shift

        return logits
