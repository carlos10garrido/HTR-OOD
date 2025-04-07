# Michael et. al, 2019
import torch
import torch.nn as nn
from src.data.components.tokenizers import Tokenizer
import torch.nn.utils.parametrize as P
import math

# Based on the previous experimental results, we chose hybrid monotonic attention with the normalized Bahdanau scoring function 
# in combination with gradient clipping and the implementation details of Sec. IV as our final architecture.
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

      
class ContentBasedAttention(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(hidden_size))
        
    def forward(self, encoder_outputs, decoder_h_t):
        # Bahdanau scoring function
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        
        query = decoder_h_t.permute(1, 0, 2).contiguous() # [B, 1, dec_dim]
        keys = encoder_outputs
        energy = self.v(torch.tanh(self.W_h(query) + self.W_s(keys) + self.b)) # [B, T, 1]
        weights = self.softmax(energy.squeeze(-1)).unsqueeze(1) # [B, 1, T]
        context = torch.bmm(weights, keys) # [B, 1, enc_dim] 
        
        return context
        

class CRNN_Michael(nn.Module):
    """ Hybrid architecture from 'Evaluating Sequence-to-Sequence Models for Handwritten Text Recognition' """

    def __init__(
        self, 
        image_size=(64, 1024),
        att_dim=128,
        input_size=128,
        hidden_size=128,
        char_embedding_size=64,
        dropout_encoder=0.5,
        layers_encoder=3,
        layers_decoder=2,
        tokenizer: Tokenizer=None
    ) -> None:

      super(CRNN_Michael, self).__init__()
      self.image_size = image_size
      self.att_dim = att_dim
      self.vocab_size = tokenizer.vocab_size
      self.tokenizer = tokenizer
      self.layers_encoder = layers_encoder
      self.layers_decoder = layers_decoder
      self.hidden_size = hidden_size

      # CNN encoder
      # The encoder is a three-layer CNN with interleaved maxpooling layers, followed by three BLSTM layers:
      self.cnn_encoder = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=(6, 4), stride=(4, 2), padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(8, 32, kernel_size=(6, 4), stride=(1, 1), padding=0),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(4, 2), stride=(4, 2)),
        nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1,1), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)),  
      )

      # Calculate image reduction factor with the pooling layers and kernel sizes
      self.img_reduction = (32, 9)
      print(f'IMAGE REDUCTION FACTOR: {self.img_reduction}')

      # Initialize convolutional layers with Xavier initialization
      for layer in self.cnn_encoder:
        if isinstance(layer, nn.Conv2d):
          nn.init.xavier_uniform_(layer.weight)
      
      self.rnn_encoder = nn.LSTM(
          input_size=input_size,
          hidden_size=self.hidden_size,
          num_layers=layers_encoder,
          bidirectional=True,
          dropout=dropout_encoder,
          batch_first=True
        )
      
      self.pe_cross = PositionalEncoding(self.hidden_size, dropout=0.1, max_len=400)

      # Linear projection (original paper implemented as a 1x1 convolutional layer
      # Linear layer is faster since it does not require permutation of the tensor
      self.ctc_pred = nn.Linear(self.hidden_size*2, self.vocab_size+1) # +1 for ctc blank token. These are H in the paper
      self.combine_c_h = nn.Linear(char_embedding_size + self.hidden_size, char_embedding_size + self.hidden_size, bias=True)
      self.act = nn.Identity()
      
      self.attention = ContentBasedAttention(self.hidden_size)
      self.conv1d = nn.Linear(self.vocab_size+1, self.hidden_size) # Conv1d layer for attention
      self.char_embedding = nn.Embedding(self.vocab_size, char_embedding_size)
      
      # Fully connected layer for context
      self.fc_context = nn.Sequential(
        nn.Linear(self.hidden_size*2, self.hidden_size),
        nn.ReLU()
      )

      self.decoder = nn.LSTM(
        input_size=char_embedding_size + self.hidden_size,
        hidden_size=self.hidden_size, # double the size of the encoder hidden size
        num_layers=layers_decoder,
        dropout=dropout_encoder,
        bidirectional=False,
        batch_first=True
      )

      # self.output = nn.Linear(hidden_size, self.vocab_size)
      self.output  = torch.nn.Sequential(  
          torch.nn.Linear(hidden_size, self.vocab_size) # +1 for CTC blank token
      )
      
      # Xavier initialization for all parameters
      for p in self.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
      
        

    def init_x(self, batch_size):
      return torch.ones((batch_size, 1), dtype=torch.long, requires_grad=False) * self.tokenizer.bos_id

    def init_h(self, batch_size, decoder_dim, num_layers=1):
      # return torch.Tensor(batch_size, num_layers, decoder_dim).normal_()
      return torch.Tensor(batch_size, num_layers, decoder_dim).zero_().requires_grad_()
    
    # Training forward
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)
      cnn_output = self.ctc_pred(x)
      x = self.conv1d(cnn_output)
      x = self.pe_cross(x)
      encoder_outputs = x#.clone()

      # Decoder 
      batch_size, dec_sequence_length = y.size()

      # Layers_decoder = 1 for this particular model
      h = self.init_h(x.shape[0], self.hidden_size, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous().requires_grad_()
      c = torch.zeros(x.shape[0], self.layers_decoder, self.hidden_size).to(x.device).permute(1, 0, 2).contiguous().requires_grad_()
      # logit_list = []
      
      logit_list = torch.zeros((batch_size, dec_sequence_length, self.vocab_size), dtype=torch.float32, requires_grad=True).to(x.device)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      # Introduce teacher forcing noise with a 10% probability in y for tokens != <bos>, <eos>, <pad>
      token_mask = (y != self.tokenizer.bos_id) & (y != self.tokenizer.eos_id) & (y != self.tokenizer.pad_id)
      noise_mask = torch.rand_like(token_mask, dtype=torch.float) <= 0.2
      y = torch.where(noise_mask & token_mask, torch.randint_like(y, 4, self.vocab_size), y) # 3 is the UNK token
      char_embeddings = self.char_embedding(y)
      
      for i in range(dec_sequence_length):
          char_embed = char_embeddings[:, i].unsqueeze(1)
          
          # Attention 
          context = self.attention(encoder_outputs, h) # Since we have to squeeze the 1 dimension
          context = self.act(self.fc_context(torch.cat([context, h.permute(1, 0, 2)], dim=-1)))
          
          input_embed = self.combine_c_h(torch.cat([char_embed, context], dim=-1))#.permute(1, 0, 2)
          out, (h, c) = self.decoder(input_embed, (h, c))
          
          # [batch_size, vocab_size]
          logit = self.output(out).squeeze(0) # Predict the next character
          logit_list[:, i] = logit.squeeze(1)

      return cnn_output.log_softmax(-1).permute(1, 0, 2), logit_list

    # Inference forward
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)
      cnn_output = self.ctc_pred(x)
      x = self.conv1d(cnn_output)
      x = self.pe_cross(x)
      encoder_outputs = x
      
      # Decoder
      batch_size, enc_sequence_length, enc_dim = x.size()

      output = self.init_x(batch_size).to(x.device)
      h = self.init_h(x.shape[0], self.hidden_size, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c = torch.zeros(x.shape[0], self.layers_decoder, self.hidden_size).to(x.device).permute(1, 0, 2).contiguous()
      output_list, raw_preds = [], []
      
      raw_preds = torch.zeros((batch_size, 120, self.vocab_size), dtype=torch.float32, requires_grad=False).to(x.device)
      output_list = torch.zeros((batch_size, 120), dtype=torch.long).to(x.device)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      for i in range(120):
          char_embed = self.char_embedding(output)

          # Attention 
          context = self.attention(encoder_outputs, h) # Since we have to squeeze the 1 dimension
          context = self.act(self.fc_context(torch.cat([context, h.permute(1, 0, 2)], dim=-1)))
          
          input_embed = self.combine_c_h(torch.cat([char_embed, context], dim=-1))#.permute(1, 0, 2)
          out, (h, c) = self.decoder(input_embed, (h, c))
          
          # [batch_size, vocab_size]
          logit = self.output(out).squeeze(0) # Predict the next character
          raw_preds[:, i] = logit.squeeze(1)

          # Greedy Decoding
          logit = torch.argmax(logit, dim=-1)
          output = logit
          output_list[:, i] = logit.squeeze(1)
          
      return output_list.detach(), raw_preds.detach()

if __name__ == "__main__":
    _ = CRNN_Michael()