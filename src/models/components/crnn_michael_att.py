# Michael et. al, 2019
import torch
import torch.nn as nn
from src.data.components.tokenizers import Tokenizer
import torch.nn.utils.parametrize as P

# Based on the previous experimental results, we chose hybrid monotonic attention with the normalized Bahdanau scoring function 
# in combination with gradient clipping and the implementation details of Sec. IV as our final architecture.

      
class ContentBasedAttention(nn.Module):
    def __init__(self, hidden_size=128):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        
        self.W_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_s = nn.Linear(hidden_size, hidden_size, bias=True)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        
    def forward(self, encoder_outputs, decoder_h_t):
        # query: decoder hidden state [B, dec_dim, 1] (1 is layer)
        # key: encoder hidden states [B, T, enc_dim]
        
        # Bahdanau scoring function
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        
        query = decoder_h_t.permute(1, 0, 2).contiguous() # [B, 1, dec_dim]
        keys = encoder_outputs
        
        energy = self.v(torch.tanh(self.W_h(query) + self.W_s(keys))) # [B, T, 1]
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

      # Linear projection (original paper implemented as a 1x1 convolutional layer
      # Linear layer is faster since it does not require permutation of the tensor
      self.ctc_pred = nn.Linear(self.hidden_size*2, self.vocab_size+1) # +1 for ctc blank token. These are H in the paper
      self.combine_c_h = nn.Linear(self.vocab_size+1 + self.hidden_size, self.hidden_size, bias=True)
      self.act = nn.Tanh()
      # self.act = nn.ReLU()
      # self.act = nn.Identity()
      
      self.attention = ContentBasedAttention(self.hidden_size)
      self.conv1d = nn.Linear(self.hidden_size*2, self.hidden_size) # Conv1d layer for attention
      self.char_embedding = nn.Embedding(self.vocab_size, char_embedding_size)

      self.decoder = nn.LSTM(
        input_size=char_embedding_size + self.hidden_size,
        hidden_size=self.hidden_size, # double the size of the encoder hidden size
        num_layers=layers_decoder,
        dropout=dropout_encoder,
        bidirectional=False,
        batch_first=True
      )

      self.output = nn.Linear(hidden_size, self.vocab_size)

    def init_x(self, batch_size):
      return torch.ones((batch_size, 1), dtype=torch.long) * self.tokenizer.bos_id

    def init_h(self, batch_size, decoder_dim, num_layers=1):
      # return torch.Tensor(batch_size, num_layers, decoder_dim).normal_()
      return torch.Tensor(batch_size, num_layers, decoder_dim).zero_()
    
    # Training forward
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
      # breakpoint()
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)
      cnn_output = self.ctc_pred(x)
      x = self.conv1d(x)
            
      encoder_outputs = x.clone()

      # Decoder 
      batch_size, enc_sequence_length, enc_dim = x.size()
      batch_size, dec_sequence_length = y.size()

      # Layers_decoder = 1 for this particular model
      h = self.init_h(1, self.hidden_size, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c = torch.zeros(1, self.layers_decoder, self.hidden_size).to(x.device).permute(1, 0, 2).contiguous()
      logit_list = []

      # logit_list.append(torch.ones(batch_size, self.vocab_size).to(x.device) * self.tokenizer.bos_id)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      char_embeddings = self.char_embedding(y)
      
      for i in range(dec_sequence_length):
          char_embed = char_embeddings[:, i].unsqueeze(1)
          # print(f'CHAR EMBED (training): {char_embed.shape}')

          # Attention 
          context = self.attention(encoder_outputs, h) # Since we have to squeeze the 1 dimension
          
          input_embed = torch.cat([char_embed, context], dim=-1).permute(1, 0, 2)
          out, (h, c) = self.decoder(input_embed, (h, c))
          
          # [batch_size, vocab_size]
          logit = self.output(out).squeeze(0) # Predict the next character
          logit_list.append(logit)

      return cnn_output.log_softmax(-1).permute(1, 0, 2), torch.stack(logit_list, dim=1)
      # return torch.stack(logit_list, dim=1) # Only for seq2seq training


    # Inference forward
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)
      cnn_output = self.ctc_pred(x)
      x = self.conv1d(x)
            
      encoder_outputs = x.clone()
      
      # Decoder
      batch_size, enc_sequence_length, enc_dim = x.size()

      output = self.init_x(batch_size).to(x.device)
      h = self.init_h(1, self.hidden_size, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c = torch.zeros(1, self.layers_decoder, self.hidden_size).to(x.device).permute(1, 0, 2).contiguous()
      output_list, raw_preds = [], []
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      # output = [batch_size, 1]
      # breakpoint()
      for i in range(120):
          char_embed = self.char_embedding(output)
          # print(f'CHAR EMBED (inference): {char_embed.shape}')

          # Attention 
          context = self.attention(encoder_outputs, h) # Since we have to squeeze the 1 dimension
          
          input_embed = torch.cat([char_embed, context], dim=-1).permute(1, 0, 2)
          out, (h, c) = self.decoder(input_embed, (h, c))
          
          # [batch_size, vocab_size]
          logit = self.output(out).squeeze(0) # Predict the next character
          raw_preds.append(logit)

          # Greedy Decoding
          logit = torch.argmax(logit, dim=-1)
          output = logit.unsqueeze(1) # [batch_size] - [batch_size, 1]
          output_list.append(logit)
          
      # print(f'VAL torch.stack(output_list, dim=1): {torch.stack(output_list, dim=1).shape}')
      return torch.stack(output_list, dim=1), torch.stack(raw_preds, dim=1)
      # return torch.stack(output_list, dim=1) # Only for seq2seq inference

if __name__ == "__main__":
    _ = CRNN_Michael()




    

