import torch
from torch import nn

class CharClassifier(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        vocab_size: int,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.proj_channels = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        self.backbone = torch.nn.Sequential(
            *list(self.backbone.children())[:-1], # Remove last layer
        )
        # Add linear layer for regression
        self.linear = torch.nn.Linear(512, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.
        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.proj_channels(x)
        x = self.backbone(x)
        x = x.flatten(1)
        return self.linear(x)
    

if __name__ == "__main__":
    _ = CharClassifier()