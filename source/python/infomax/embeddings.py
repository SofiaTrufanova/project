import torch
import torchkld

from . import channels


class Embedder(torch.nn.Module):
    def __init__(self, embedder_network: callable, discriminator_network: torchkld.mutual_information.MINE,
                 input_channel: channels.Channel, output_channel: channels.Channel,
                 detach: bool=False) -> None:
        
        super().__init__()
        self.embedder_network = embedder_network
        self.discriminator_network = discriminator_network

        self.input_channel = input_channel
        self.output_channel = output_channel

        self.detach = detach

    def forward(self, x, marginalize: bool=True):       
        embeddings = self.output_channel(self.embedder_network(x))

        if self.detach:
            with torch.no_grad():
                noisy_embeddings = self.embedder_network(self.input_channel(x)).detach()
        else:
            noisy_embeddings = self.embedder_network(self.input_channel(x))

        T_joined   = self.discriminator_network(embeddings, noisy_embeddings)
        T_marginal = self.discriminator_network(embeddings, noisy_embeddings, marginalize=marginalize)

        return T_joined, T_marginal