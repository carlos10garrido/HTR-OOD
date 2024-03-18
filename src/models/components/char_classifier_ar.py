import torch
from torch import nn

class CharClassifierAR(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        vocab_size: int,
    ) -> None:
        """Initialize the model."""
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
        self.proj_channels = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0)
        # self.backbone = torch.nn.Sequential(
        #     *list(self.backbone.children())[:-4], # Remove last layer
        # )
        self.backbone = torch.nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.ReLU(inplace=True),
          nn.AvgPool2d(kernel_size=2, stride=2),
          nn.InstanceNorm2d(16),
          nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.ReLU(inplace=True),
          nn.AvgPool2d(kernel_size=2, stride=2),
          nn.InstanceNorm2d(32),
          nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
          nn.ReLU(inplace=True),
          nn.AvgPool2d(kernel_size=2, stride=2),
          nn.InstanceNorm2d(32),
        )



        self.lin_proj_to_lstm = nn.Linear(32*8*16, input_size)
        # Add bi-lstm for predicting char patches from segmentation
        self.lstm = torch.nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          bidirectional=True
        )
        
        # Add linear layer for classification
        self.out  = torch.nn.Sequential(  
          torch.nn.Linear(2048, 1024),
          torch.nn.ReLU(),
          torch.nn.Linear(1024, vocab_size)
        )
        # torch.nn.Linear(2048, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.
        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        B, S, C, H, W = x.shape
        print(f'x.shape: {x.shape} before proj_channels')
        x = x.view(B*S, C, H, W)
        x = self.proj_channels(x)
        x = self.backbone(x)
        print(f'x.shape: {x.shape} after backbone')
        x = x.reshape(B, S, -1)
        print(f'x.shape: {x.shape} after reshape')
        x = self.lin_proj_to_lstm(x)
        # x, _ = self.lstm(x.unsqueeze(0))
        x, _ = self.lstm(x)
        print(f'x.shape: {x.shape} after lstm')
        # x = x.flatten(1)
        return self.out(x)
    

if __name__ == "__main__":
    _ = CharClassifierAR()