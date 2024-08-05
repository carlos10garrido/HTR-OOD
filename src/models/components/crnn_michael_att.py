# Michael et. al, 2019
import torch
import torch.nn as nn
from src.data.components.tokenizers import Tokenizer
import torch.nn.utils.parametrize as P

# Based on the previous experimental results, we chose hybrid monotonic attention with the normalized Bahdanau scoring function 
# in combination with gradient clipping and the implementation details of Sec. IV as our final architecture.

      
class ContentBasedAttention(nn.Module):
    def __init__(self, enc_dim=128, dec_dim=128, att_dim=128):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        
        self.W_h = nn.Linear(dec_dim, att_dim, bias=False)
        self.W_s = nn.Linear(enc_dim, att_dim, bias=False)
        self.v = nn.Linear(att_dim, 1, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())
        
    def forward(self, encoder_outputs, decoder_h_t):
        # Bahdanau scoring function
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        
        # [batch_size, sequence_length, att_dim]
        energy = torch.tanh(self.W_h(decoder_h_t).unsqueeze(1).repeat(1, sequence_length, 1) +
                            self.W_s(encoder_outputs) + self.b)
        energy = self.v(energy).squeeze(-1)
        att_weights = self.softmax(energy)
        
        return att_weights # [B, T] containing the attention weights for that timestep
        

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
          hidden_size=input_size,
          num_layers=layers_encoder,
          bidirectional=True,
          dropout=dropout_encoder,
          batch_first=True
        )

      # Linear projection (original paper implemented as a 1x1 convolutional layer
      # Linear layer is faster since it does not require permutation of the tensor
      self.ctc_pred = nn.Linear(2*input_size, self.vocab_size+1) # +1 for ctc blank token
      self.combine_c_h = nn.Linear(hidden_size*4, hidden_size, bias=True)
      # self.act = nn.Tanh()
      self.act = nn.ReLU()
      
      self.attention = ContentBasedAttention(input_size*2, input_size*2, att_dim)
      self.char_embedding = nn.Embedding(self.vocab_size, char_embedding_size)

      self.decoder = nn.LSTM(
        input_size=char_embedding_size, # character embedding size
        hidden_size=hidden_size*2, # double the size of the encoder hidden size
        num_layers=layers_decoder,
        dropout=dropout_encoder,
        bidirectional=False,
        batch_first=True
      )

      self.output = nn.Linear(hidden_size, self.vocab_size)

    def init_x(self, batch_size):
      return torch.ones((batch_size, 1), dtype=torch.long) * self.tokenizer.bos_id

    def init_h(self, batch_size, decoder_dim, num_layers=1):
      return torch.Tensor(batch_size, num_layers, decoder_dim).normal_()
      # return torch.Tensor(batch_size, num_layers, decoder_dim).zero_()
    
    

    # Training forward
    def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)

      # Return x for CTC loss
      encoder_outputs = x.clone()
      cnn_output = self.ctc_pred(x)

      # Decoder 
      batch_size, enc_sequence_length, enc_dim = x.size()
      batch_size, dec_sequence_length = y.size()

      output = self.init_x(batch_size).to(x.device)
      h = self.init_h(batch_size, enc_dim, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c_0 = torch.zeros(batch_size, self.layers_decoder, enc_dim).to(x.device).permute(1, 0, 2).contiguous()
      context = c_0
      logit_list = []
      # append B, 1 with zeros to logit_list
      logit_list.append(torch.ones(batch_size, self.vocab_size).to(x.device) * self.tokenizer.bos_id)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      for i in range(dec_sequence_length):
          output = self.char_embedding(output)
          x, (h, context) = self.decoder(output, (h, context))
          
          # Attention 
          alpha = self.attention(encoder_outputs, h[0])

          # Weighted-sum
          # [batch_size, out_dim]
          context = torch.sum(alpha.unsqueeze(-1) * encoder_outputs, dim=1)
          
          concat_c_h = torch.cat([context, h[0]], dim=-1)
          attentional = self.act(self.combine_c_h(concat_c_h)).reshape(batch_size, -1)

          # [batch_size, vocab_size]
          logit = self.output(attentional)
          # print(f'Logit shape: {logit.shape}')
          logit_list.append(logit)

          output = y[:, i] # Teacher forcing
          output = output.unsqueeze(1)
          
          context = context.unsqueeze(0)
          # print(f'Output shape in teacher forcing: {output.shape}')
          
      # print(f'torch.stack(logit_list, dim=1): {torch.stack(logit_list, dim=1).shape}')

      # return cnn_output.log_softmax(-1).permute(1, 0, 2), torch.stack(logit_list, dim=1)
      return torch.stack(logit_list, dim=1) # Only for seq2seq training


    # Inference forward
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
      # CNN encoder
      x = self.cnn_encoder(x)
      x = x.flatten(1, 2).permute(0, 2, 1)
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)
      encoder_outputs = x.clone()

      
      # Decoder
      batch_size, enc_sequence_length, enc_dim = x.size()

      output = self.init_x(batch_size).to(x.device)
      h = self.init_h(batch_size, enc_dim, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c_0 = torch.zeros(batch_size, self.layers_decoder, enc_dim).to(x.device).permute(1, 0, 2).contiguous()
      context = c_0
      output_list = []
      output_list.append(torch.ones(batch_size).to(x.device) * self.tokenizer.bos_id)
      
      raw_preds = []
      
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      for i in range(120):
          output = self.char_embedding(output)
          x, (h, context) = self.decoder(output, (h, context))

          # Attention
          alpha = self.attention(encoder_outputs, h[0])

          # Weighted-sum
          # [batch_size, out_dim]
          context = torch.sum(alpha.unsqueeze(-1) * encoder_outputs, dim=1)
          attentional = self.act(self.combine_c_h(torch.cat([context, h[0]], dim=-1)).reshape(batch_size, -1))

          # [batch_size, vocab_size]
          logit = self.output(attentional)
          raw_preds.append(logit)

          # Greedy Decoding
          logit = torch.argmax(logit, dim=-1)
          output = logit.unsqueeze(1)
          # print(f'Output shape in greedy decoding: {output.shape}')
          output_list.append(logit)
          
          context = context.unsqueeze(0)

      # print(f'VAL torch.stack(output_list, dim=1): {torch.stack(output_list, dim=1).shape}')
      return torch.stack(output_list, dim=1), torch.stack(raw_preds, dim=1)
      # return torch.stack(output_list, dim=1) # Only for seq2seq inference

if __name__ == "__main__":
    _ = CRNN_Michael()




    


      

