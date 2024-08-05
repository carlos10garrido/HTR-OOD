# CNN-SAN Arce et. al, 2022
import torch
import torch.nn as nn
from src.data.components.tokenizers import Tokenizer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        # pe = pe.squeeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: torch.Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.unsqueeze(-2) + self.pe[:x.size(1)]
        x = self.dropout(x)
        x = x.squeeze(-2)
        return x

class ConvEncoder(nn.Module):
  def __init__(self, dropout: float = 0.2) -> None:
      super(ConvEncoder, self).__init__()
      # Is the same as the original CRNN backbone (Puigcerver et al., 2017, ICDAR)
      self.backbone = torch.nn.Sequential(
        ### Block 1
        nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
        nn.BatchNorm2d(16), 
        nn.LeakyReLU(inplace=True, negative_slope=0.001),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ### Block 2
        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
        nn.BatchNorm2d(32), 
        nn.LeakyReLU(inplace=True, negative_slope=0.001),
        nn.MaxPool2d(kernel_size=2, stride=2),
        ### Block 3
        nn.Conv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
        nn.BatchNorm2d(48), 
        nn.LeakyReLU(inplace=True, negative_slope=0.001),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Dropout2d(dropout),
        ### Block 4
        nn.Conv2d(48, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
        nn.BatchNorm2d(64), 
        nn.LeakyReLU(inplace=True, negative_slope=0.001),
        nn.Dropout2d(dropout),
        ### Block 5
        nn.Conv2d(64, 80, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False), 
        nn.BatchNorm2d(80), 
        nn.LeakyReLU(inplace=True, negative_slope=0.001),
        nn.Dropout2d(dropout),
      )
    
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      return self.backbone(x)
      

class CNN_SAN_Arce(nn.Module):
    def __init__(
      self, 
      cnn_dropout: float = 0.2,
      dropout: float = 0.4, 
      n_heads: int = 4,
      n_layers: int = 4,
      d_model: int = 192,
      hidden_dim: int = 512,
      img_size: tuple = (32, 128),
      tokenizer: Tokenizer = None
    ) -> None:
      
      super(CNN_SAN_Arce, self).__init__()
      self.cnn_encoder = ConvEncoder(cnn_dropout)
      self.tokenizer = tokenizer
      self.img_reduction = (13, 9)
      self.vocab_size = tokenizer.vocab_size
      # self.pe_encoder = PositionalEncoding(d_model)
      self.embedding_encoder = nn.Linear(1280, d_model)
      self.self_attention = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=n_heads,
          dim_feedforward=hidden_dim,
          dropout=dropout,
          activation='relu',
          norm_first=False,
        ),
        num_layers=n_layers
      )
      
      self.class_head = nn.Linear(d_model, self.vocab_size+1) # +1 for CTC blank token
      
      # Initialize all weights with Glorot Uniform
      for p in self.parameters():
          if p.dim() > 1:
              nn.init.xavier_uniform_(p)

    # Training forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f'Image shape: {x.shape}')
        x = self.cnn_encoder(x)
        # print(f'CNN output shape: {x.shape}')
        x = x.flatten(1, 2).permute(0, 2, 1)
        x = self.embedding_encoder(x)
        # x = self.pe_encoder(x)        
        x = self.self_attention(x)
        x = self.class_head(x)

        return x

    
if __name__ == "__main__":
    _ = CNN_SAN_Arce()



    


      

