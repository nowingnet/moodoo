import torch

class DiscriminatorModel(torch.nn.Module):
    def __init__(self, dim_latent=32, dim_expressions=76):
        super(DiscriminatorModel, self).__init__()

        self.model = torch.nn.Sequential(
            torch.nn.Linear(dim_latent, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_latent*2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(dim_latent*2, dim_expressions),
            torch.nn.Tanh(),
        )

    def forward(self, x):

        return self.model(x)