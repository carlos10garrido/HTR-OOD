import torch
from torch import nn

# import tokenizers
from src.data.components.tokenizers import Tokenizer

class CRNN_Puig(nn.Module):
  def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        # vocab_size: int,
        tokenizer: Tokenizer
    ) -> None:
        """Initialize the model."""
        super().__init__()
        # Original CRNN backbone (Puigcerver et al., 2017, ICDAR)
        self.backbone = torch.nn.Sequential(
          ### Block 1
          nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
          nn.BatchNorm2d(16), 
          nn.LeakyReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          ### Block 2
          nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
          nn.BatchNorm2d(32), 
          nn.LeakyReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          ### Block 3
          nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
          nn.BatchNorm2d(48), 
          nn.LeakyReLU(inplace=True),
          nn.MaxPool2d(kernel_size=2, stride=2),
          nn.Dropout2d(0.2),
          ### Block 4
          nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
          nn.BatchNorm2d(64), nn.LeakyReLU(inplace=True),
          nn.Dropout2d(0.2),
          ### Block 5
          nn.Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
          nn.BatchNorm2d(80), 
          nn.LeakyReLU(inplace=True),
          nn.Dropout2d(0.2),
        )

        self.img_reduction = (32, 8)
        self.vocab_size = tokenizer.vocab_size
        self.hidden_size = hidden_size

        # Add bi-lstm for predicting char patches from segmentation
        self.lstm = torch.nn.LSTM( # 5 layers for Puigcerver et al., 2017
          input_size=input_size,
          hidden_size=hidden_size,
          num_layers=num_layers,
          batch_first=True,
          bidirectional=True,
          dropout=0.5
        )
        
        # Add linear layer for classification
        self.out  = torch.nn.Sequential(  
          nn.Dropout(0.5),
          torch.nn.Linear(self.hidden_size*2, self.vocab_size+1) # +1 for CTC blank token
        )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
      """Perform a single forward pass through the network.
      :param x: The input tensor.
      :return: A tensor of predictions.
      """

      B, C, H, W = x.shape
      x = self.backbone(x)
      # print(f'x.shape: {x.shape} after backbone')
      x = x.view(B, x.shape[1]*x.shape[2], x.shape[3]).permute(0, 2, 1)
      # print(f'x.shape: {x.shape} after view and permute')
      self.lstm.flatten_parameters()
      # breakpoint()
      x, _ = self.lstm(x)
      # print(f'x.shape: {x.shape} after lstm')
      return self.out(x)
    

if __name__ == "__main__":
    _ = CRNN_Puig()