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
            
class CNN_Encoder(nn.Module):
  """CNN Encoder for the Light Barrere architecture"""
  
  def __init__(self, input_shape):
      super(CNN_Encoder, self).__init__()
      self.input_shape = input_shape
      print(f'input_shape: {input_shape} to LN (first)')
        
      # Separte through layers to get the output
      # Formula to calculate the output shape of a convolutional layer
      # output_shape = [(input_shape - kernel_size + 2*padding)/stride] + 1
      # Block 1
      self.conv1 = nn.Conv2d(3, 8, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
      self.leaky_relu = nn.LeakyReLU()
      input_shape = (8, int((input_shape[1]-3+2*0)/1+1), int((input_shape[2]-3+2*0)/1+1))
      print(f'input_shape: {input_shape} to LN')
      
      self.layer_norm1 = nn.LayerNorm(input_shape, elementwise_affine=False)
      self.max_pool1 = nn.MaxPool2d(kernel_size=(2, 2))
      input_shape = (8, input_shape[1]//2, input_shape[2]//2)
      self.dropout = nn.Dropout(0.2)
      
      # Block 2
      self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
      
      input_shape = (16, int((input_shape[1]-3+2*0)/1+1), int((input_shape[2]-3+2*0)/1+1))
      print(f'input_shape: {input_shape} to LN')
      self.layer_norm2 = nn.LayerNorm(input_shape, elementwise_affine=False)
      self.max_pool2 = nn.MaxPool2d(kernel_size=(2, 2))
      input_shape = (16, input_shape[1]//2, input_shape[2]//2)
      
      # Block 3
      self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
      input_shape = (32, int((input_shape[1]-3+2*0)/1+1), int((input_shape[2]-3+2*0)/1+1))
      print(f'input_shape: {input_shape} to LN')
      self.layer_norm3 = nn.LayerNorm(input_shape, elementwise_affine=False)
      self.max_pool3 = nn.MaxPool2d(kernel_size=(2, 2))
      input_shape = (32, input_shape[1]//2, input_shape[2]//2)
      
      # Block 4
      self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
      input_shape = (64, int((input_shape[1]-3+2*0)/1+1), int((input_shape[2]-3+2*0)/1+1))
      print(f'input_shape: {input_shape} to LN')
      self.layer_norm4 = nn.LayerNorm(input_shape, elementwise_affine=False)
      
      # Block 5
      self.conv5 = nn.Conv2d(64, 128, kernel_size=(4, 2), stride=(1, 1), padding=(0, 0))
      input_shape = (128, int((input_shape[1]-4+2*0)/1+1), int((input_shape[2]-2+2*0)/1+1))
      print(f'input_shape: {input_shape} to LN')
      self.layer_norm5 = nn.LayerNorm(input_shape, elementwise_affine=False)
        
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Block 1
    x = self.conv1(x)
    x = self.leaky_relu(x)
    x = self.layer_norm1(x)
    x = self.max_pool1(x)
    x = self.dropout(x)
    
    # Block 2
    x = self.conv2(x)
    x = self.leaky_relu(x)
    x = self.layer_norm2(x)
    x = self.max_pool2(x)
    x = self.dropout(x)
    
    # Block 3
    x = self.conv3(x)
    x = self.leaky_relu(x)
    x = self.layer_norm3(x)
    x = self.max_pool3(x)
    x = self.dropout(x)
    
    # Block 4
    x = self.conv4(x)
    x = self.leaky_relu(x)
    x = self.layer_norm4(x)
    x = self.dropout(x)
    
    # Block 5
    x = self.conv5(x)
    x = self.leaky_relu(x)
    x = self.layer_norm5(x)
    x = self.dropout(x)
    
    return x

    

class Light_Barrere(nn.Module):
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
        char_embedding_size=256,
        tokenizer: Tokenizer=None
    ) -> None:

      super(Light_Barrere, self).__init__()
      self.image_size = image_size
      self.vocab_size = tokenizer.vocab_size
      self.tokenizer = tokenizer
      self.leaky_relu = nn.LeakyReLU()
      self.cnn_encoder = CNN_Encoder(input_shape=(3, image_size[0], image_size[1]))

      # Calculate image reduction factor with the pooling layers and kernel sizes
      self.img_reduction = (14, 9)
      print(f'IMAGE REDUCTION FACTOR: {self.img_reduction}')

      # Dense layer to collapse the CNN output
      
      self.collapse_layer = nn.Linear(1152, 128)
      self.layer_norm_collapse = nn.LayerNorm(128, elementwise_affine=False)
      self.dense = nn.Linear(128, hidden_dim)
      
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
          norm_first=False
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
          norm_first=False
        ),
        num_layers=decoder_layers
      )
    
      # Character embedding for the decoder
      self.char_embedding = nn.Embedding(self.vocab_size, char_embedding_size)
      
      # Output layer
      self.output = nn.Linear(hidden_dim, self.vocab_size)
      
      # Initialize weights glorot
      for p in self.parameters():
        if p.dim() > 1:
          nn.init.xavier_uniform_(p)

      

    # Training forward
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
      # CNN encoder
      cnn_output = self.cnn_encoder(x)
      cnn_output = cnn_output.flatten(1,2).permute(0, 2, 1) # [B, W, C*H]
      cnn_output = self.collapse_layer(cnn_output) # [B, W, d_model]
      cnn_output = self.leaky_relu(cnn_output)
      cnn_output = self.layer_norm_collapse(cnn_output)
      cnn_output = self.dense(cnn_output)
      
      # Transformer encoder
      pe_output = self.pe_encoder(cnn_output)
      encoder_output = self.encoder(pe_output)

      # CTC prediction # [B,W,vocab_size+1] -> (permute) -> [W ,B,vocab_size+1]
      pred_ctc_encoder = self.pred_ctc(encoder_output).log_softmax(-1).permute(1, 0, 2) 

      # Cross attention positional encoding
      cross_output = self.pe_cross(encoder_output)
      
      tgt_key_padding_mask = (y == self.tokenizer.pad_id).to(x.device)
      
      # Transformer decoder
      embedding_output = self.char_embedding(y.to(x.device).long())

      # Positional encoding for the decoder
      pe_decoder_output = self.pe_decoder(embedding_output)
      
      decoder_output = self.decoder(
        tgt=pe_decoder_output,
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(y.size(1)).to(x.device),
        memory=cross_output,
        tgt_key_padding_mask=tgt_key_padding_mask.to(x.device),
        tgt_is_causal=True
      )

      # Output layer
      output = self.output(decoder_output)

      return pred_ctc_encoder, output


    # Inference forward
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
      # Create batch size of BOS tokens
      output = torch.ones((x.size(0), 1), dtype=torch.int) * self.tokenizer.bos_id
      output = output.to(x.device)

      # CNN encoder
      cnn_output = self.cnn_encoder(x)
      cnn_output = cnn_output.flatten(1,2).permute(0, 2, 1) # [B, W, C*H]
      cnn_output = self.collapse_layer(cnn_output) # [B, W, d_model]
      cnn_output = self.leaky_relu(cnn_output)
      cnn_output = self.layer_norm_collapse(cnn_output)
      cnn_output = self.dense(cnn_output)
      
      # Transformer encoder
      pe_output = self.pe_encoder(cnn_output)
      encoder_output = self.encoder(pe_output)

      # CTC prediction # [B,W,vocab_size+1] -> (permute) -> [W ,B,vocab_size+1]
      pred_ctc_encoder = self.pred_ctc(encoder_output).permute(1, 0, 2) # Not used

      # Cross attention positional encoding
      cross_output = self.pe_cross(encoder_output)
      
      # Initialize output tensor
      output = torch.ones((x.size(0), 1), dtype=torch.int) * self.tokenizer.bos_id
      output = output.to(x.device)
      
      raw_output = []

      for i in range(1, 128):
        embedding_output = self.char_embedding(output)
        pe_decoder_output = self.pe_decoder(embedding_output)
        decoder_output = self.decoder(
          tgt=pe_decoder_output,
          memory=cross_output,
        )
        
        raw_output.append(self.output(decoder_output)[:,-1,:])
        next_token = self.output(decoder_output)[:,-1].argmax(dim=-1).unsqueeze(-1)
        output = torch.cat([output, next_token], dim=-1)

      # breakpoint()
      return output[:, 1:], torch.stack(raw_output, dim=1)

if __name__ == "__main__":
    _ = Light_Barrere()




    


      

