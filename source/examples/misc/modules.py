import math
import torch
import torchfd
import infomax


class AdditiveGaussainT(torchfd.mutual_information.MINE):
    def __init__(self, p: float=0.1, marginalizer=None) -> None:
        super().__init__(marginalizer=marginalizer)

        self.p_logit = torch.nn.Parameter(torch.logit(torch.tensor(p)), requires_grad=True)
        self.bias = 1.0 # From optimal solution for NWJ

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        p = torch.sigmoid(self.p_logit)
        p_squared = p**2
        
        result = 0.5 * (torch.sum(x**2, axis=-1) - torch.sum((x - y * torch.sqrt(1.0 - p_squared))**2, axis=-1) / p_squared) - \
                torch.nn.functional.logsigmoid(self.p_logit) + self.bias
        
        return result


class AffineAdditiveGaussainT(torchfd.mutual_information.MINE):
    def __init__(self, dim: int, p: float=0.1, marginalizer=None) -> None:
        super().__init__(marginalizer=marginalizer)

        self.p_logit = torch.nn.Parameter(torch.logit(torch.tensor(p)), requires_grad=True)
        self.bias = 1.0 # From optimal solution for NWJ

        self.linear_X = torch.nn.Linear(dim, dim)
        self.linear_Y = torch.nn.Linear(dim, dim)

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        x = self.linear_X(x)
        y = self.linear_X(y)

        p = torch.sigmoid(self.p_logit)
        p_squared = p**2
        
        result = 0.5 * (torch.sum(x**2, axis=-1) - torch.sum((x - y * torch.sqrt(1.0 - p_squared))**2, axis=-1) / p_squared) - \
                torch.nn.functional.logsigmoid(self.p_logit) + self.bias
        
        return result


class DenseT(torchfd.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=256, marginalizer=None) -> None:
        super().__init__(marginalizer=marginalizer)

        self.model = torch.nn.Sequential(
            torch.nn.Linear(X_dim + Y_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, 1)
        )        

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:
        return self.model(torch.cat((x, y), dim=1))


class SeparableT(torchfd.mutual_information.MINE):
    def __init__(self, X_dim: int, Y_dim: int, inner_dim: int=128, output_dim: int=64, marginalizer=None) -> None:
        super().__init__(marginalizer=marginalizer)
        
        self.projector_x = torch.nn.Sequential(
            torch.nn.Linear(X_dim, inner_dim),
            torch.nn.BatchNorm1d(inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, output_dim)
            #torch.nn.BatchNorm1d(output_dim),
        )
        self.projector_y = torch.nn.Sequential(
            torch.nn.Linear(Y_dim, inner_dim),
            torch.nn.BatchNorm1d(inner_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(inner_dim, output_dim),
            #torch.nn.BatchNorm1d(output_dim),
        )        

    @torchfd.mutual_information.MINE.marginalized
    def forward(self, x: torch.tensor, y: torch.tensor) -> torch.tensor:        
        # Projection.
        x = self.projector_x(x)
        y = self.projector_y(y)
        
        return torch.mean(x * y, dim=-1)


class Conv2dEmbedder(torch.nn.Module):
    """
    Convolutional embedder.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Noise.
        self.channel = infomax.channels.BoundedVarianceGaussianChannel(1.0e-3)
        
        # Activations.
        self.activation = torch.nn.LeakyReLU()
        #self.output_activation = output_activation
        
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
        
        return x #self.output_activation(x)


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