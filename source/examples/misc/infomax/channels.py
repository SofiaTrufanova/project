import torch
import math


class Channel(torch.nn.Module):
    """
    A noisy channel base class.
    """

    def __init__(self, enabled_on_inference: bool=False) -> None:
        """
        Create an instance of the `NoisyChannel` class
        """

        super().__init__()
        
        self.enabled_on_inference = enabled_on_inference

    @property
    def capacity(self):
        """ Capacity of the channel. """
        raise NotImplementedError


class BoundedVarianceGaussianChannel(Channel):
    """
    Assuming the input being of a unit variance, rescales the input and adds an
    independent white Gaussian noise in such a way that the variance of
    the output is also unit.
    """
    
    def __init__(self, p: float=0.1, enabled_on_inference: bool=False) -> None:
        """
        Create an instance of the `BoundedVarianceGaussianChannel` class
        """

        if p < 0.0 or p > 1.0:
            raise ValueError("Expected `p` to be within [0;1].")
        
        super().__init__(enabled_on_inference)
        
        self.p = p

        # A separate parameter for noise.
        # Needed in order to generate noise on the desired device.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Is preferable in case of multiple devices.

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        """

        if (self.training or self.enabled_on_inference) and self.p != 0.0:
            sampled_noise = self.noise.repeat(*x.size()).normal_()
            x = math.sqrt(1.0 - self.p**2) * x + self.p * sampled_noise
        
        return x

    @property
    def correlation_coefficient(self):
        """ Correlation between the input and the output. """
        return math.sqrt(1.0 - self.p**2)

    @property
    def capacity(self):
        """ Capacity of the channel. """
        return -math.log(self.p)


class BoundedSupportGaussianChannel(Channel):
    """
    Assuming the input being of a bounded support (in [0;1]^d),
    adds an independent white Gaussian noise.
    """
    
    def __init__(self, sigma: float=1.0e-2, enabled_on_inference: bool=False) -> None:
        """
        Create an instance of the `BoundedVarianceGaussianChannel` class
        """

        if sigma < 0.0:
            raise ValueError("Expected `sigma` to be non-negative.")
        
        super().__init__(enabled_on_inference)
        
        self.sigma = sigma

        # A separate parameter for noise.
        # Needed in order to generate noise on the desired device.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Is preferable in case of multiple devices.

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        """

        if (self.training or self.enabled_on_inference) and self.sigma != 0.0:
            sampled_noise = self.noise.repeat(*x.size()).normal_()
            x = x + self.sigma * sampled_noise
        
        return x

    @property
    def capacity(self):
        """ Asymptotic capacity of the channel. """
        return math.log(60.0 / math.pi**2) * self.sigma - 0.5 * math.log(2.0 * math.pi * math.e * self.sigma**2)


class BoundedSupportUniformChannel(Channel):
    """
    Assuming the input being of a bounded support (in [0;1]^d), rescales the
    input and adds an independent white uniform noise.
    """
    
    def __init__(self, p: float=1.0e-2, enabled_on_inference: bool=False) -> None:
        """
        Create an instance of the `BoundedVarianceGaussianChannel` class
        """

        if p < 0.0 or p > 1.0:
            raise ValueError("Expected `sigma` to be non-negative.")
        
        super().__init__(enabled_on_inference)
        
        self.p = p

        # A separate parameter for noise.
        # Needed in order to generate noise on the desired device.
        #self.noise = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        #self.noise.requires_grad_(False)
        self.register_buffer('noise', torch.tensor(0.0, dtype=torch.float32))  # Is preferable in case of multiple devices.

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward pass.
        """

        if (self.training or self.enabled_on_inference) and self.p != 0.0:
            sampled_noise = self.noise.repeat(*x.size()).uniform_()
            x = (1.0 - self.p) * x + self.p * sampled_noise
        
        return x

    @property
    def capacity(self):
        """ Capacity of the channel. """
        epsilon = 0.5 * self.p / (1.0 - self.p)
        return epsilon - math.log(2.0 * epsilon) if epsilon < 0.5 else 0.25 / epsilon