import math
import torch
import torchkld
import infomax


class AdditiveGaussainT(torchkld.mutual_information.MINE):
    def __init__(self, p: float=0.1) -> None:
        super().__init__()

        self.p_logit = torch.nn.Parameter(torch.logit(torch.tensor(p)), requires_grad=True)
        self.bias = 1.0 # From optimal solution for NWJ
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)

        p = torch.sigmoid(self.p_logit)
        p_squared = p**2
        
        result = 0.5 * (torch.sum(x**2, axis=-1) - torch.sum((x - y * torch.sqrt(1.0 - p_squared))**2, axis=-1) / p_squared) - \
                torch.nn.functional.logsigmoid(self.p_logit) + self.bias
        
        return result


class DenseT(torchkld.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=256) -> None:
        super().__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(X_dim + Y_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, 1)
        )        
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize) 
        
        return self.model(torch.cat((x, y), dim=1))


class SeparableT(torchkld.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=128, output_dim: int=64) -> None:
        super().__init__()
        
        self.projector_x = torch.nn.Sequential(
            torch.nn.Linear(X_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, output_dim)
        )
        self.projector_y = torch.nn.Sequential(
            torch.nn.Linear(Y_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, output_dim)
        )        
        
    def forward(self, x: torch.tensor, y: torch.tensor, marginalize: bool=False) -> torch.tensor:
        x, y = super().forward(x, y, marginalize)
        
        # Projection.
        x = self.projector_x(x)
        y = self.projector_y(y)
        
        return torch.mean(x * y, dim=-1)


class Conv2dEmbedder(torch.nn.Module):
    """
    Convolutional embedder.
    """

    def __init__(self, embedding_dim: int, output_activation: torch.nn.Module=torch.nn.Sigmoid()):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Noise.
        self.channel = infomax.channels.BoundedVarianceGaussianChannel(1.0e-3)
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        self.output_activation = output_activation
        
        # Convolution layers.
        self.conv2d_1 = torch.nn.Conv2d(1, 32, kernel_size=3)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=3)
        self.conv2d_3 = torch.nn.Conv2d(64, 128, kernel_size=3)
        self.pool2d = torch.nn.AvgPool2d((2,2))
        
        self.batchnorm2d_1 = torch.nn.BatchNorm2d(32)
        self.batchnorm2d_2 = torch.nn.BatchNorm2d(64)
        self.batchnorm2d_3 = torch.nn.BatchNorm2d(128)
        
        # Dense layers.
        self.linear_1 = torch.nn.Linear(128, 128)
        self.linear_2 = torch.nn.Linear(128, self.embedding_dim)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # Convolution layer №1
        #x = self.dropout(x)
        x = self.conv2d_1(x)
        x = self.pool2d(x)
        x = self.activation(x)
        x = self.batchnorm2d_1(x)
        #x = self.channel(x)
        
        # Convolution layer №2
        #x = self.dropout(x)
        x = self.conv2d_2(x)
        x = self.pool2d(x)
        x = self.activation(x)
        x = self.batchnorm2d_2(x)
        #x = self.channel(x)
        
        # Convolution layer №3
        #x = self.dropout(x)
        x = self.conv2d_3(x)
        x = self.pool2d(x)
        x = self.activation(x)
        x = self.batchnorm2d_3(x)
        #x = self.channel(x)
        
        # Dense layer №1
        x = self.linear_1(torch.flatten(x, 1))
        x = self.activation(x)
        #x = self.channel(x)
        
        # Dense layer №2
        x = self.linear_2(x)
        
        return self.output_activation(x)


class ResidualBlock(torch.nn.Module):
    """
    Residual-блок с суммой в конце.
    """
    
    def __init__(self, n_channels: int=32, n_convolutions: int=2,
                 conv2d_args: dict={"kernel_size": 3, "stride": 1, "padding": "same", "padding_mode": "reflect"},
                 activation: torch.nn.Module=torch.nn.LeakyReLU()):
        super().__init__()

        self.convolutions = torch.nn.ModuleList([torch.nn.Conv2d(n_channels, n_channels, **conv2d_args) for index in range(n_convolutions)])
        self.activation = activation

    def forward(self, x: torch.tensor) -> torch.tensor:
        y = x
        for convolution in self.convolutions:
            y = convolution(y)
            y = self.activation(y)

        return x + y


class ResNetEmbedder(torch.nn.Module):
    """
    Дискриминатор на residual-блоках.
    Можно как подавать метку класса, так и не подавать (см. конструктор).
    """
    
    def __init__(self, input_shape: tuple, n_channels_list: list, latent_dim: int, embedding_dim: int,
                 output_activation: torch.nn.Module=torch.nn.Sigmoid(),
                 bottleneck_conv2d_args: dict={"kernel_size": 1, "stride": 1, "padding": "same", "padding_mode": "reflect"},
                 pooling: torch.nn.Module=torch.nn.AvgPool2d(kernel_size=2, ceil_mode=True),
                 activation: torch.nn.Module=torch.nn.LeakyReLU()) -> None:
        super().__init__()

        self.input_shape  = input_shape
        self.latent_dim   = latent_dim
        self.embedding_dim = embedding_dim

        self.output_activation = output_activation

        self.activation = activation
        self.pooling = pooling

        self.residual_blocks = torch.nn.ModuleList([ResidualBlock(n_channels=n_channels) for n_channels in n_channels_list[1:]])
        self.bottleneck_convolutions = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(in_channels=n_channels_list[index], out_channels=n_channels_list[index+1], **bottleneck_conv2d_args) for index in range(len(n_channels_list)-1)
            ]
        )
        self.batch_normalizations = torch.nn.ModuleList([torch.nn.BatchNorm2d(n_channels) for n_channels in n_channels_list[1:]])

        self.postprocessing = torch.nn.Sequential(
            torch.nn.Linear(self.latent_dim, self.latent_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(self.latent_dim, self.embedding_dim),
        )

    def forward(self, images: torch.tensor) -> torch.tensor:
        x = images
        
        for index in range(len(self.residual_blocks)):
            x = self.bottleneck_convolutions[index](x)
            x = self.batch_normalizations[index](x)
            x = self.activation(x)
            x = self.residual_blocks[index](x)

            x = self.pooling(x)

        x = torch.flatten(x, 1)
        x = self.postprocessing(x)

        return self.output_activation(x)


class DenseClassifier(torch.nn.Module):
    def __init__(self, input_dim, n_classes, device, n_layers: int=1, hidden_dim: int=2048, n_epochs=2000):
        super().__init__()

        self.device = device
        
        if n_layers > 1:
            layers = [torch.nn.Linear(input_dim, hidden_dim), torch.nn.LeakyReLU()]
            
            for layer_index in range(n_layers-2):
                layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
                layers.append(torch.nn.LeakyReLU())
            
            layers.append(torch.nn.Linear(hidden_dim, n_classes))
            
            self.model = torch.nn.Sequential(*layers)
        else:
            self.model = torch.nn.Linear(input_dim, n_classes)

        self.n_epochs = n_epochs
        self.loss = torch.nn.CrossEntropyLoss()

    def fit(self, x, y):
        x = torch.tensor(x, device=self.device)
        y = torch.tensor(y, device=self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1.0e-3)

        for epoch in range(self.n_epochs):
            optimizer.zero_grad()
            
            logits = self.model(x)
            loss_value = self.loss(logits, y)
            loss_value.backward()
            
            optimizer.step()

    def predict_proba(self, x):
        was_in_trainig = self.model.training
        self.model.eval()

        x = torch.tensor(x, device=self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            y_pred = torch.nn.functional.softmax(logits, dim=-1)

        self.model.train(was_in_trainig)

        return y_pred.cpu().detach().numpy()