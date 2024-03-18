import torch
from torch import nn


class CRNN(nn.Module):
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

        # Original CRNN backbone (Shi et al., 2015)
        self.backbone = torch.nn.Sequential(
          nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(inplace=True),
          nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
          nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), nn.ReLU(inplace=True),
          nn.BatchNorm2d(512),
          nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
          nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1), padding=(1, 1), bias=False),  nn.ReLU(inplace=True),
        )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # self.lin_proj_to_lstm = nn.Linear(32*8, input_size)
        # Add bi-lstm for predicting char patches from segmentation
        self.lstm = torch.nn.LSTM(
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          bidirectional=True
        )

        self.img_reduction = 8
        
        # Add linear layer for classification
        self.out  = torch.nn.Sequential(  
          torch.nn.Linear(self.hidden_size*2, self.hidden_size*2),
          torch.nn.ReLU(),
          torch.nn.Linear(self.hidden_size*2, vocab_size+1) # +1 for CTC blank token
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Perform a single forward pass through the network.
      :param x: The input tensor.
      :return: A tensor of predictions.
      """
      # breakpoint()
      B, C, H, W = x.shape
      x = self.proj_channels(x) if C == 1 else x
      x = self.backbone(x)
      x = x.view(B, x.shape[1]*x.shape[2], x.shape[3]).permute(0, 2, 1)
      # x = self.lin_proj_to_lstm(x)
      x, _ = self.lstm(x)
      return self.out(x)
    

if __name__ == "__main__":
    _ = CRNN()