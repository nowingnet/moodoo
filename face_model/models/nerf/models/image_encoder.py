
import torch
class ImageEncoder(torch.nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.n_down = 5
        # Bx3x256x256 -> Bx128x1x1
        self.cnn_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 64

            torch.nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 16

            torch.nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 4

            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2), # 1

            torch.nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0),
            torch.nn.Tanh()
        )

    def forward(self,x):
        x = self.cnn_layers(x)
        return x