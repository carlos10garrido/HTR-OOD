import torch
from torch import nn

class ResnetRegression(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.backbone = torch.nn.Sequential(
            *list(self.backbone.children())[:-1], # Remove last layer
        )
        # Add linear layer for regression
        self.linear = torch.nn.Linear(512, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.
        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.backbone(x)
        x = x.flatten(1)
        print(f'x.shape: {x.shape} after flatten')
        return self.linear(x)
    

if __name__ == "__main__":
    _ = ResnetRegression()