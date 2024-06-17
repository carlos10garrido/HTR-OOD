# K. Barrere et. al, 2022
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

class V_Light_Barrere(nn.Module):
    """ Hybrid architecture from 'A Light Transformer-Based Architecture for Handwritten Text Recognition' """

    def __init__(
        self, 
        image_size=(64, 1024),
        hidden_dim=256,
        intermediate_ffn_dim=1024,
        dropout=0.1,
        n_heads=4,
        encoder_layers=4,
        decoder_layers=4,
        input_size=128,
        char_embedding_size=256,
        tokenizer: Tokenizer=None
    ) -> None:

      super(V_Light_Barrere, self).__init__()
      self.image_size = image_size
      self.vocab_size = tokenizer.vocab_size
      self.tokenizer = tokenizer


      # CNN encoder
      # Except the last one, each convolutional block is composed of a 2D convolutional layer with a kernel of size 3×3, a stride of 1 and no padding. 
      # The last convolutional block uses a kernel size of 4×2 to better match the shape of a character [3,13]. 
      # The number of filters in the convolutional layers are respectively equal to 8, 16, 32, 64 and 128. 
      # Each convolution layer is then followed by a LeakyReLU activation function. 
      # Following the activation function, we apply a layer normalization to ease the network 
      # training capabilities and increase the regularization capacities of the network. 
      # A 2 × 2 max pooling is used inside the first three convolutional blocks to decrease the size of intermediate feature maps. 
      # Lastly, a dropout is applied with a probability of 0.2 at the end of each block.
      # Convert the comments to the actual code
      self.cnn_encoder = nn.Sequential(
        # Convolutional block 1
        nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        nn.LeakyReLU(),
        nn.LayerNorm((8, 126, 1022), elementwise_affine=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout(0.1),

        # Convolutional block 2
        nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        nn.LeakyReLU(),
        nn.LayerNorm((16, 61, 509), elementwise_affine=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout(0.1),

        # Convolutional block 3
        nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        nn.LeakyReLU(),
        nn.LayerNorm((32, 28, 252), elementwise_affine=False),
        nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        nn.Dropout(0.1),

        # Convolutional block 4
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        nn.LeakyReLU(),
        nn.LayerNorm((64, 12, 124), elementwise_affine=False),
        nn.Dropout(0.1),

        # Convolutional block 5
        nn.Conv2d(64, 128, kernel_size=(4, 2), stride=(1, 1), padding=(0, 0)),
        nn.LeakyReLU(),
        nn.LayerNorm((128, 9, 123), elementwise_affine=False),
        nn.Dropout(0.1),
          
      )

      # Calculate image reduction factor with the pooling layers and kernel sizes
      self.img_reduction = (14, 9)
      print(f'IMAGE REDUCTION FACTOR: {self.img_reduction}')

      # Dense layer to collapse the CNN output
      
      self.collapse_layer = nn.Linear(1152, hidden_dim)
      self.pred_ctc = nn.Linear(hidden_dim, self.vocab_size+1) # +1 for CTC blank token

      self.pe_encoder = PositionalEncoding(hidden_dim)
      self.pe_cross = PositionalEncoding(hidden_dim)
      self.pe_decoder = PositionalEncoding(char_embedding_size)

      self.encoder = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
          d_model=hidden_dim,
          nhead=n_heads,
          dim_feedforward=intermediate_ffn_dim,
          dropout=dropout,
          batch_first=True,
          # bias=False,
          norm_first=True
        ),
        num_layers=encoder_layers
      )

      self.decoder = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
          d_model=hidden_dim,
          nhead=n_heads,
          dim_feedforward=intermediate_ffn_dim,
          dropout=dropout,
          batch_first=True,
          # bias=False,
          norm_first=True
        ),
        num_layers=decoder_layers
      )
    
      # Character embedding for the decoder
      self.char_embedding = nn.Embedding(self.vocab_size, char_embedding_size)
      
      # Output layer
      self.output = nn.Linear(hidden_dim, self.vocab_size)

      

    # Training forward
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:

      # breakpoint()
      # CNN encoder
      # print(f'x.shape: {x.shape}')
      cnn_output = self.cnn_encoder(x)
      # print(f'cnn_output.shape: {cnn_output.shape}')
      cnn_output = cnn_output.flatten(1,2).permute(0, 2, 1) # [B, W, C*H]
      cnn_output = self.collapse_layer(cnn_output) # [B, W, d_model]
      
      # Transformer encoder
      pe_output = self.pe_encoder(cnn_output)
      encoder_output = self.encoder(pe_output)

      # CTC prediction # [B,W,vocab_size+1] -> (permute) -> [W ,B,vocab_size+1]
      pred_ctc_encoder = self.pred_ctc(encoder_output).permute(1, 0, 2) 

      # Cross attention positional encoding
      cross_output = self.pe_cross(encoder_output)

      # Add BOS token to the target
      y = torch.cat([torch.ones(y.shape[0], 1, dtype=torch.int).to(x.device) * self.tokenizer.bos_id, y], dim=-1)
      tgt_padding_mask = y == self.tokenizer.pad_id

      # Transformer decoder
      embedding_output = self.char_embedding(y)

      # Positional encoding for the decoder
      pe_decoder_output = self.pe_decoder(embedding_output)
      
      decoder_output = self.decoder(
        tgt=pe_decoder_output,
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(y.size(1)),
        memory=cross_output,
        tgt_key_padding_mask=tgt_padding_mask,
        tgt_is_causal=True
      )

      # Output layer
      output = self.output(decoder_output)

      return pred_ctc_encoder, output


    # Inference forward
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
      # breakpoint()
      # Create batch size of BOS tokens
      output = torch.ones((x.size(0), 1), dtype=torch.int) * self.tokenizer.bos_id
      output = output.to(x.device)

      # CNN encoder
      # print(f'x.shape: {x.shape}')
      cnn_output = self.cnn_encoder(x)
      # print(f'cnn_output.shape: {cnn_output.shape}')
      cnn_output = cnn_output.flatten(1,2).permute(0, 2, 1) # [B, W, C*H]
      cnn_output = self.collapse_layer(cnn_output) # [B, W, d_model]
      
      # Transformer encoder
      pe_output = self.pe_encoder(cnn_output)
      encoder_output = self.encoder(pe_output)

      # CTC prediction # [B,W,vocab_size+1] -> (permute) -> [W ,B,vocab_size+1]
      pred_ctc_encoder = self.pred_ctc(encoder_output).permute(1, 0, 2) 

      # Cross attention positional encoding
      cross_output = self.pe_cross(encoder_output)


      for i in range(1, 150):
        embedding_output = self.char_embedding(output)
        pe_decoder_output = self.pe_decoder(embedding_output)
        decoder_output = self.decoder(
          tgt=pe_decoder_output,
          memory=cross_output,
        )
        # output = torch.cat([output, self.output(decoder_output).argmax(dim=-1)[:, -1].unsqueeze(-1)], dim=1)

        # next_token = self.output(decoder_output)[:,-1,:].argmax(dim=-1).unsqueeze(-1)
        next_token = self.output(decoder_output)[:,-1].argmax(dim=-1).unsqueeze(-1)
        output = torch.cat([output, next_token], dim=-1)

      # breakpoint()
      return output

if __name__ == "__main__":
    _ = V_Light_Barrere()




    


      

