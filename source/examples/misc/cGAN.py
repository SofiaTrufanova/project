"""
Follows https://github.com/kwotsin/mimicry
"""

import torch
import torch_mimicry

import math

class FullConditionalBatchNorm2d(torch.nn.Module):
    r"""
    Conditional Batch Norm as implemented in
    https://github.com/pytorch/pytorch/issues/8985

    Attributes:
        num_features (int): Size of feature map for batch norm.
        condition_dim (int): Determines size of embedding layer to condition BN.
    """
    def __init__(self, num_features, condition_dim):
        super().__init__()
        self.num_features = num_features
        self.bn = torch.nn.BatchNorm2d(num_features, affine=False)
        self.condition_linear = torch.nn.Linear(condition_dim, num_features * 2)

        self.condition_linear.bias.data[:num_features].data.normal_(1, 0.02)
        self.condition_linear.bias.data[num_features:].data.zero_()

        self.condition_linear.weight.data.normal_(0, 0.02 / math.sqrt(num_features))
        
        #self.embed = nn.Embedding(condition_dim, num_features * 2)
        #self.embed.weight.data[:, :num_features].normal_(
        #    1, 0.02)  # Initialise scale at N(1, 0.02)
        #self.embed.weight.data[:,
        #                       num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        r"""
        Feedforwards for conditional batch norm.

        Args:
            x (Tensor): Input feature map.
            y (Tensor): Input class labels for embedding.

        Returns:
            Tensor: Output feature map.
        """
        out = self.bn(x)
        gamma, beta = self.condition_linear(y).chunk(
            2, 1)  # divide into 2 chunks, split from dim 1.
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(
            -1, self.num_features, 1, 1)

        return out



import torch.nn as nn
import torch.nn.functional as F

from torch_mimicry.modules import SNConv2d, ConditionalBatchNorm2d


class GBlock(nn.Module):
    r"""
    Residual block for generator.

    Uses bilinear (rather than nearest) interpolation, and align_corners
    set to False. This is as per how torchvision does upsampling, as seen in:
    https://github.com/pytorch/vision/blob/master/torchvision/models/segmentation/_utils.py

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        upsample (bool): If True, upsamples the input feature map.
        condition_dim (int): If more than 0, uses conditional batch norm instead.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 upsample=False,
                 condition_dim=0,
                 spectral_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else out_channels
        self.learnable_sc = in_channels != out_channels or upsample
        self.upsample = upsample

        self.condition_dim = condition_dim
        self.spectral_norm = spectral_norm

        # Build the layers
        # Note: Can't use something like self.conv = SNConv2d to save code length
        # this results in somehow spectral norm working worse consistently.
        if self.spectral_norm:
            #self.c1 = SNConv2d(self.in_channels,
            #                   self.hidden_channels,
            #                   3,
            #                   1,
            #                   padding=1)
            #self.c2 = SNConv2d(self.hidden_channels,
            #                   self.out_channels,
            #                   3,
            #                   1,
            #                   padding=1)
            self.c1 = torch.nn.utils.spectral_norm(nn.Conv2d(self.in_channels,
                                self.hidden_channels,
                                3,
                                1,
                                padding=1,
                                padding_mode="reflect"))
            self.c2 = torch.nn.utils.spectral_norm(nn.Conv2d(self.hidden_channels,
                                self.out_channels,
                                3,
                                1,
                                padding=1,
                                padding_mode="reflect"))
        else:
            self.c1 = nn.Conv2d(self.in_channels,
                                self.hidden_channels,
                                3,
                                1,
                                padding=1,
                                padding_mode="reflect")
            self.c2 = nn.Conv2d(self.hidden_channels,
                                self.out_channels,
                                3,
                                1,
                                padding=1,
                                padding_mode="reflect")

        if self.condition_dim == 0:
            self.b1 = nn.BatchNorm2d(self.in_channels)
            self.b2 = nn.BatchNorm2d(self.hidden_channels)
        else:
            self.b1 = FullConditionalBatchNorm2d(self.in_channels,
                                             self.condition_dim)
            self.b2 = FullConditionalBatchNorm2d(self.hidden_channels,
                                             self.condition_dim)

        self.activation = nn.LeakyReLU(inplace=True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                self.c_sc = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels,
                                      out_channels,
                                      1,
                                      1,
                                      padding=0,
                                      padding_mode="reflect"))
            else:
                self.c_sc = nn.Conv2d(in_channels,
                                      out_channels,
                                      1,
                                      1,
                                      padding=0,
                                      padding_mode="reflect")

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _upsample_conv(self, x, conv):
        r"""
        Helper function for performing convolution after upsampling.
        """
        return conv(
            F.interpolate(x,
                          scale_factor=2,
                          mode='bilinear',
                          align_corners=False))

    def _residual(self, x):
        r"""
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.b1(h)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _residual_conditional(self, x, y):
        r"""
        Helper function for feedforwarding through main layers, including conditional BN.
        """
        h = x
        h = self.b1(h, y)
        h = self.activation(h)
        h = self._upsample_conv(h, self.c1) if self.upsample else self.c1(h)
        h = self.b2(h, y)
        h = self.activation(h)
        h = self.c2(h)

        return h

    def _shortcut(self, x):
        r"""
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self._upsample_conv(
                x, self.c_sc) if self.upsample else self.c_sc(x)
            return x
        else:
            return x

    def forward(self, x, y=None):
        r"""
        Residual block feedforward function.
        """
        if y is None:
            return self._residual(x) + self._shortcut(x)

        else:
            return self._residual_conditional(x, y) + self._shortcut(x)


class DBlock(nn.Module):
    """
    Residual block for discriminator.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        hidden_channels (int): The channel size of intermediate feature maps.
        downsample (bool): If True, downsamples the input feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden_channels=None,
                 downsample=False,
                 spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels if hidden_channels is not None else in_channels
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            #self.c1 = SNConv2d(self.in_channels, self.hidden_channels, 3, 1, 1)
            #self.c2 = SNConv2d(self.hidden_channels, self.out_channels, 3, 1,
            #                   1)
            self.c1 = torch.nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1, padding_mode="reflect"))
            self.c2 = torch.nn.utils.spectral_norm(nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, 1, 1, padding_mode="reflect"))
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.hidden_channels, 3, 1, 1, padding_mode="reflect")
            self.c2 = nn.Conv2d(self.hidden_channels, self.out_channels, 3, 1, 1, padding_mode="reflect")

        self.activation = nn.LeakyReLU(inplace=True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))

        # Shortcut layer
        if self.learnable_sc:
            if self.spectral_norm:
                #self.c_sc = SNConv2d(in_channels, out_channels, 1, 1, 0)
                self.c_sc = torch.nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, 1, 1, 0, padding_mode="reflect"))
            else:
                self.c_sc = nn.Conv2d(in_channels, out_channels, 1, 1, 0, padding_mode="reflect")

            nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        if self.learnable_sc:
            x = self.c_sc(x)
            return F.avg_pool2d(x, 2) if self.downsample else x

        else:
            return x

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)


class DBlockOptimized(nn.Module):
    """
    Optimized residual block for discriminator. This is used as the first residual block,
    where there is a definite downsampling involved. Follows the official SNGAN reference implementation
    in chainer.

    Attributes:
        in_channels (int): The channel size of input feature map.
        out_channels (int): The channel size of output feature map.
        spectral_norm (bool): If True, uses spectral norm for convolutional layers.        
    """
    def __init__(self, in_channels, out_channels, spectral_norm=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spectral_norm = spectral_norm

        # Build the layers
        if self.spectral_norm:
            #self.c1 = SNConv2d(self.in_channels, self.out_channels, 3, 1, 1)
            #self.c2 = SNConv2d(self.out_channels, self.out_channels, 3, 1, 1)
            #self.c_sc = SNConv2d(self.in_channels, self.out_channels, 1, 1, 0)
            self.c1 = torch.nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"))
            self.c2 = torch.nn.utils.spectral_norm(nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect"))
            self.c_sc = torch.nn.utils.spectral_norm(nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, padding_mode="reflect"))
        else:
            self.c1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1, padding_mode="reflect")
            self.c2 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1, padding_mode="reflect")
            self.c_sc = nn.Conv2d(self.in_channels, self.out_channels, 1, 1, 0, padding_mode="reflect")

        self.activation = nn.LeakyReLU(True)

        nn.init.xavier_uniform_(self.c1.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c2.weight.data, math.sqrt(2.0))
        nn.init.xavier_uniform_(self.c_sc.weight.data, 1.0)

    def _residual(self, x):
        """
        Helper function for feedforwarding through main layers.
        """
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = F.avg_pool2d(h, 2)

        return h

    def _shortcut(self, x):
        """
        Helper function for feedforwarding through shortcut layers.
        """
        return self.c_sc(F.avg_pool2d(x, 2))

    def forward(self, x):
        """
        Residual block feedforward function.
        """
        return self._residual(x) + self._shortcut(x)



import torch
import torch.nn as nn

#from torch_mimicry.nets.cgan_pd import cgan_pd_base
from torch_mimicry.modules import SNLinear, SNEmbedding
#from torch_mimicry.modules.resblocks import DBlockOptimized, DBlock, GBlock


from torch_mimicry.nets.gan import gan


class BaseConditionalGenerator(gan.BaseGenerator):
    r"""
    Base class for a generic conditional generator model.

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, condition_dim, nz, ngf, bottom_width, loss_type,
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         **kwargs)
        self.condition_dim = condition_dim

    def generate_images(self, num_images, c=None, device=None):
        r"""
        Generate images with possibility for conditioning on a fixed class.

        Args:
            num_images (int): The number of images to generate.
            c (int): The class of images to generate. If None, generates random images.
            device (int): The device to send the generated images to.

        Returns:
            tuple: Batch of generated images and their corresponding labels.
        """
        if device is None:
            device = self.device

        if c is None:
            fake_class_labels = torch.normal(size=(num_images, self.condition_dim), device=device)

        else:
            fake_class_labels = c

        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise, fake_class_labels)

        return fake_images

    def generate_images_with_labels(self, num_images, c=None, device=None):
        r"""
        Generate images with possibility for conditioning on a fixed class.
        Additionally returns labels.

        Args:
            num_images (int): The number of images to generate.
            c (int): The class of images to generate. If None, generates random images.
            device (int): The device to send the generated images to.

        Returns:
            tuple: Batch of generated images and their corresponding labels.
        """
        if device is None:
            device = self.device

        if c is None:
            fake_class_labels = torch.randn(size=(num_images, self.condition_dim), device=device)

        else:
            fake_class_labels = c

        noise = torch.randn((num_images, self.nz), device=device)
        fake_images = self.forward(noise, fake_class_labels)

        return fake_images, fake_class_labels

    def train_step(self,
                   real_batch,
                   netD,
                   optG,
                   #log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for G.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
                Used for obtaining current batch size.
            netD (nn.Module): Discriminator model for obtaining losses.
            optG (Optimizer): Optimizer for updating generator's parameters.
            log_data (MetricLog): A dict mapping name to values for logging uses.
            device (torch.device): Device to use for running the model.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.

        """
        self.zero_grad()

        # Get only batch size from real batch
        batch_size = real_batch[0].shape[0]

        # Produce fake images and labels
        fake_images, fake_class_labels = self.generate_images_with_labels(
            num_images=batch_size, device=device)

        # Compute output logit of D thinking image real
        output = netD(fake_images, fake_class_labels)

        # Compute loss and backprop
        errG = self.compute_gan_loss(output)

        # Backprop and update gradients
        errG.backward()
        optG.step()

        # Log statistics
        #log_data.add_metric('errG', errG, group='loss')

        return errG #log_data


class BaseConditionalDiscriminator(gan.BaseDiscriminator):
    r"""
    Base class for a generic conditional discriminator model.

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, condition_dim, ndf, loss_type, **kwargs):
        super().__init__(ndf=ndf, loss_type=loss_type, **kwargs)
        self.condition_dim = condition_dim

    def train_step(self,
                   real_batch,
                   netG,
                   optD,
                   #log_data,
                   device=None,
                   global_step=None,
                   **kwargs):
        r"""
        Takes one training step for D.

        Args:
            real_batch (Tensor): A batch of real images of shape (N, C, H, W).
            loss_type (str): Name of loss to use for GAN loss.
            netG (nn.Module): Generator model for obtaining fake images.
            optD (Optimizer): Optimizer for updating discriminator's parameters.
            device (torch.device): Device to use for running the model.
            log_data (MetricLog): A dict mapping name to values for logging uses.
            global_step (int): Variable to sync training, logging and checkpointing.
                Useful for dynamic changes to model amidst training.

        Returns:
            MetricLog: Returns MetricLog object containing updated logging variables after 1 training step.
        """
        self.zero_grad()

        real_images, real_class_labels = real_batch
        batch_size = real_images.shape[0]  # Match batch sizes for last iter

        # Produce logits for real images
        output_real = self.forward(real_images, real_class_labels)

        # Produce fake images and labels
        fake_images, fake_class_labels = netG.generate_images_with_labels(
            num_images=batch_size, device=device)
        fake_images, fake_class_labels = fake_images.detach(
        ), fake_class_labels.detach()

        # Produce logits for fake images
        output_fake = self.forward(fake_images, fake_class_labels)

        # Compute loss for D
        errD = self.compute_gan_loss(output_real=output_real,
                                     output_fake=output_fake)

        # Backprop and update gradients
        errD.backward()
        optD.step()

        # Compute probabilities
        D_x, D_Gz = self.compute_probs(output_real=output_real,
                                       output_fake=output_fake)

        # Log statistics for D once out of loop
        #log_data.add_metric('errD', errD, group='loss')
        #log_data.add_metric('D(x)', D_x, group='prob')
        #log_data.add_metric('D(G(z))', D_Gz, group='prob')

        return errD, D_x, D_Gz


class CGANPDBaseGenerator(BaseConditionalGenerator):
    r"""
    ResNet backbone generator for cGAN-PD,

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self,
                 condition_dim,
                 bottom_width,
                 nz,
                 ngf,
                 loss_type='hinge',
                 **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         loss_type=loss_type,
                         condition_dim=condition_dim,
                         **kwargs)


class CGANPDBaseDiscriminator(BaseConditionalDiscriminator):
    r"""
    ResNet backbone discriminator for cGAN-PD.

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, condition_dim, ndf, loss_type='hinge', **kwargs):
        super().__init__(ndf=ndf,
                         loss_type=loss_type,
                         condition_dim=condition_dim,
                         **kwargs)


class CGANPDGenerator32(CGANPDBaseGenerator):
    r"""
    ResNet backbone generator for cGAN-PD,

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.    
        nz (int): Noise dimension for upsampling.
        ngf (int): Variable controlling generator feature map sizes.
        bottom_width (int): Starting width for upsampling generator output to an image.
        loss_type (str): Name of loss to use for GAN loss.        
    """
    def __init__(self, condition_dim, bottom_width=4, nz=128, ngf=256, **kwargs):
        super().__init__(nz=nz,
                         ngf=ngf,
                         bottom_width=bottom_width,
                         condition_dim=condition_dim,
                         **kwargs)

        # Build the layers
        self.l1 = nn.Linear(self.nz + self.condition_dim, (self.bottom_width**2) * self.ngf)
        self.block2 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             condition_dim=self.condition_dim)
        self.block3 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             condition_dim=self.condition_dim)
        self.block4 = GBlock(self.ngf,
                             self.ngf,
                             upsample=True,
                             condition_dim=self.condition_dim)
        self.b5 = nn.BatchNorm2d(self.ngf)
        self.c5 = nn.Conv2d(self.ngf, 3, 3, 1, padding=1, padding_mode="reflect")
        self.activation = nn.LeakyReLU(inplace=True)

        # Initialise the weights
        nn.init.xavier_uniform_(self.l1.weight.data, 1.0)
        nn.init.xavier_uniform_(self.c5.weight.data, 1.0)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of noise vectors into a batch of fake images, also
        conditioning the batch norm with labels of the images to be produced.

        Args:
            x (Tensor): A batch of noise vectors of shape (N, nz).
            y (Tensor): A batch of labels of shape (N,) for conditional batch norm.

        Returns:
            Tensor: A batch of fake images of shape (N, C, H, W).
        """
        if y is None:
            y = torch.randn(size=(x.shape[0], self.condition_dim), device=x.device)

        h = self.l1(torch.cat([x, y], dim=-1))
        h = h.view(x.shape[0], -1, self.bottom_width, self.bottom_width)
        h = self.block2(h, y)
        h = self.block3(h, y)
        h = self.block4(h, y)
        h = self.b5(h)
        h = self.activation(h)
        h = self.c5(h)
        h = 2*torch.tanh(h)

        return h


class CGANPDDiscriminator32(CGANPDBaseDiscriminator):
    r"""
    ResNet backbone discriminator for cGAN-PD.

    Attributes:
        condition_dim (int): Number of classes, more than 0 for conditional GANs.        
        ndf (int): Variable controlling discriminator feature map sizes.
        loss_type (str): Name of loss to use for GAN loss.                
    """
    def __init__(self, condition_dim, ndf=128, **kwargs):
        super().__init__(ndf=ndf, condition_dim=condition_dim, **kwargs)

        # Build layers
        self.block1 = DBlockOptimized(3, self.ndf)
        self.block2 = DBlock(self.ndf, self.ndf, downsample=True)
        self.block3 = DBlock(self.ndf, self.ndf, downsample=False)
        self.block4 = DBlock(self.ndf, self.ndf, downsample=False)
        self.l5 = torch.nn.utils.spectral_norm(torch.nn.Linear(self.ndf, 1))
        #self.l5 = torch.nn.Linear(self.ndf, 1)

        # Produce label vector from trained embedding
        #self.l_y = SNEmbedding(num_embeddings=self.condition_dim,
        #                       embedding_dim=self.ndf)
        self.l_y = torch.nn.utils.spectral_norm(torch.nn.Linear(self.condition_dim, self.ndf))

        # Initialise the weights
        nn.init.xavier_uniform_(self.l5.weight.data, 1.0)
        nn.init.xavier_uniform_(self.l_y.weight.data, 1.0)

        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x, y=None):
        r"""
        Feedforwards a batch of real/fake images and produces a batch of GAN logits.
        Further projects labels to condition on the output logit score.

        Args:
            x (Tensor): A batch of images of shape (N, C, H, W).
            y (Tensor): A batch of labels of shape (N,).

        Returns:
            output (Tensor): A batch of GAN logits of shape (N, 1).
        """
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)

        # Global sum pooling
        h = torch.sum(h, dim=(2, 3))
        output = self.l5(h)

        # Add the projection loss
        w_y = self.l_y(y)
        output += torch.sum((w_y * h), dim=1, keepdim=True)

        return output