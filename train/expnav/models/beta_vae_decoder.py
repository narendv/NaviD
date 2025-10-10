import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
# from torch import tensor as Tensor
Tensor = TypeVar('torch.tensor')


class BetaVAEDecoder(nn.Module):

    num_iter = 0 # Global static variable to keep track of iterations

    def __init__(self,
                 latent_dim: int,
                 input_dim: List,
                 hidden_dims: List = None,
                 beta: int = 4,
                 gamma:float = 1000.,
                 max_capacity: int = 25,
                 Capacity_max_iter: int = 1e5,
                 loss_type:str = 'B',
                 **kwargs) -> None:
        super(BetaVAEDecoder, self).__init__()

        self.latent_dim = latent_dim
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter

        modules = []
        if hidden_dims is None:
            hidden_dims = [latent_dim // (2 ** i) for i in range(4)]
            hidden_dims += [64 if hidden_dims[-1] > 64 else 32]
            hidden_dims.reverse()
        print(f"Decoder hidden dims: {hidden_dims}")

        flattened_size = hidden_dims[-1] * 3 * 3
        self.project = nn.Sequential(
            nn.Linear(input_dim, flattened_size),
            nn.LeakyReLU(inplace=True),
        )
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc_mu = nn.Linear(hidden_dims[-1]*3*3, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*3*3, latent_dim)


        # Build Decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*3*3)
        hidden_dims.reverse()
        
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            # Removed nn.Sigmoid() - now outputs logits
                            )

    def get_gaussians(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        input = self.project(input)
        mu = self.fc_mu(input)
        log_var = self.fc_var(input)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, self.latent_dim, 3, 3)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Will a single z be enough ti compute the expectation
        for the loss??
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.get_gaussians(input)
        z = self.reparameterize(mu, log_var)
        return {
            "rgb_2": self.decode(z),
            "mu": mu,
            "log_var": log_var,
        }

    # def loss_function(self,
    #                   *args,
    #                   **kwargs) -> dict:
    #     self.num_iter += 1
    #     recons = args[0]
    #     input = args[1]
    #     mu = args[2]
    #     log_var = args[3]
    #     kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset

    #     recons_loss =F.mse_loss(recons, input)

    #     kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

    #     if self.loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
    #         loss = recons_loss + self.beta * kld_weight * kld_loss
    #     elif self.loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
    #         self.C_max = self.C_max.to(input.device)
    #         C = torch.clamp(self.C_max/self.C_stop_iter * self.num_iter, 0, self.C_max.data[0])
    #         loss = recons_loss + self.gamma * kld_weight* (kld_loss - C).abs()
    #     else:
    #         raise ValueError('Undefined loss type.')

    #     return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
    

def kl_gaussian_standard_normal(mu, logvar, reduction="mean"):
    # KL(q(z|x) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    if reduction == "mean":
        return kl.mean()
    elif reduction == "sum":
        return kl.sum()
    else:
        return kl  # no reduction

def recon_loss_l1(x_hat, x, reduction="mean"):
    return F.l1_loss(x_hat, x, reduction=reduction)

def binary_cross_entropy(x_hat, x, reduction="mean"):
    # Use binary_cross_entropy_with_logits since model now outputs logits
    return F.binary_cross_entropy_with_logits(x_hat, x, reduction=reduction)

def beta_vae_loss(x_hat, x, mu, logvar, beta=1.0, recon_type="bce"):
    if recon_type == "l1":
        rec = recon_loss_l1(x_hat, x, reduction="mean")
    elif recon_type == "bce":
        rec = binary_cross_entropy(x_hat, x, reduction="mean")
    else:
        raise ValueError("recon_type must be 'l1' or 'bce'")
    kl  = kl_gaussian_standard_normal(mu, logvar, reduction="mean")
    loss = rec + beta * kl     # this is the negative ELBO to minimize
    return loss, rec, kl

if __name__ == "__main__":
    from torchinfo import summary

    model = BetaVAEDecoder(input_dim=1280, latent_dim=768)
    w = torch.randn((2, 1280))
    out = model(w)
    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: {v.shape}")
        else:
            print(f"{k}: {v}")

    summary(model, input_size=(2, 1280))