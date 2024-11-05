# Michael et. al, 2019
import torch
import torch.nn as nn
from src.data.components.tokenizers import Tokenizer
import torch.nn.utils.parametrize as P

# Based on the previous experimental results, we chose hybrid monotonic attention with the normalized Bahdanau scoring function 
# in combination with gradient clipping and the implementation details of Sec. IV as our final architecture.

# class SqrtParametrization(nn.Module): # torch 2.1
#     def __init__(self, att_dim):
#         super().__init__()
#         self.att_dim = att_dim
#     def forward(self, input):
#         return 1 / self.att_dim * input.sqrt()


class Energy(nn.Module):
    def __init__(self, enc_dim=128, dec_dim=128, att_dim=128, init_r=-4):
        super().__init__()
        self.tanh = nn.Tanh()
        self.W = nn.Linear(enc_dim, att_dim, bias=False)
        self.V = nn.Linear(dec_dim, att_dim, bias=False)
        self.b = nn.Parameter(torch.Tensor(att_dim).normal_())

        self.v = nn.utils.parametrizations.weight_norm(nn.Linear(att_dim, 1, bias=False), name='weight')
        # self.v = P.register_parametrization( # torch 2.1
        #   module=nn.Linear(att_dim, 1, bias=False), 
        #   tensor_name='weight', 
        #   parametrization=SqrtParametrization(att_dim)
        # )
        self.g = nn.Parameter(torch.Tensor([1 / att_dim]).sqrt())
        print(f'Conent of v: {self.v}')
        # self.v.weight_g.data = torch.Tensor([1 / att_dim]).sqrt()
        self.r = nn.Parameter(torch.Tensor([init_r]))

    def forward(self, encoder_outputs, decoder_h):
        # breakpoint()
        batch_size, sequence_length, enc_dim = encoder_outputs.size()
        # encoder_outputs = encoder_outputs.reshape(-1, enc_dim)
        energy = self.tanh(self.W(encoder_outputs) +
                           self.V(decoder_h).permute(1, 0, 2) + #.repeat(1, sequence_length, 1) +
                           self.b)
        
        energy = self.g * self.v(energy).squeeze(-1) + self.r # Normalized Bahdanau scoring function

        return energy.reshape(batch_size, sequence_length)
      
      
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
        

class MonotonicAttention(nn.Module):
    def __init__(self, enc_dim=128, dec_dim=128, att_dim=128, init_r=0):
        super().__init__()
        self.monotonic_energy = Energy(enc_dim*2, dec_dim*2, att_dim, init_r)
        self.sigmoid = nn.Sigmoid()

    def gaussian_noise(self, *size):
        if torch.cuda.is_available():
            return torch.cuda.FloatTensor(*size).normal_()
        else:
            return torch.Tensor(*size).normal_()

    def safe_cumprod(self, x):
        # tf.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
        return torch.exp(torch.cumsum(torch.log(torch.clamp(x, min=1e-10, max=1.0)), dim=1))

    def exclusive_cumprod(self, x):
        batch_size, sequence_length = x.size()
        if torch.cuda.is_available():
            one_x = torch.cat([torch.ones(batch_size, 1).cuda(), x], dim=1)[:, :-1]
        else:
            one_x = torch.cat([torch.ones(batch_size, 1), x], dim=1)[:, :-1]
        return torch.cumprod(one_x, dim=1)

    def soft(self, encoder_outputs, decoder_h, previous_alpha=None):
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
        p_select = self.sigmoid(monotonic_energy + self.gaussian_noise(monotonic_energy.size()))
        cumprod_1_minus_p = self.safe_cumprod(1 - p_select)

        if previous_alpha is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            alpha = torch.zeros(batch_size, sequence_length)
            alpha[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available:
                alpha = alpha.cuda()

        else:
            alpha = p_select * cumprod_1_minus_p * torch.cumsum(previous_alpha / torch.clamp(cumprod_1_minus_p, min=1e-10, max=1.0), dim=1)

        return alpha

    def hard(self, encoder_outputs, decoder_h, previous_attention=None):
        batch_size, sequence_length, enc_dim = encoder_outputs.size()

        if previous_attention is None:
            # First iteration => alpha = [1, 0, 0 ... 0]
            attention = torch.zeros(batch_size, sequence_length)
            attention[:, 0] = torch.ones(batch_size)
            if torch.cuda.is_available:
                attention = attention.cuda()
        else:
            monotonic_energy = self.monotonic_energy(encoder_outputs, decoder_h)
            # Hard Sigmoid
            # Attend when monotonic energy is above threshold (Sigmoid > 0.5)
            # above_threshold = (monotonic_energy > 0).float() # BEFORE
            above_threshold = (self.sigmoid(monotonic_energy) > 0.5).float() # NOW
            p_select = above_threshold * torch.cumsum(previous_attention, dim=1)
            attention = p_select * self.exclusive_cumprod(1 - p_select)

            # Not attended => attend at last encoder output
            # Assume that encoder outputs are not padded
            attended = attention.sum(dim=1)
            for batch_i in range(batch_size):
                if not attended[batch_i]:
                    attention[batch_i, -1] = 1

            # Ex)
            # p_select                        = [0, 0, 0, 1, 1, 0, 1, 1]
            # 1 - p_select                    = [1, 1, 1, 0, 0, 1, 0, 0]
            # exclusive_cumprod(1 - p_select) = [1, 1, 1, 1, 0, 0, 0, 0]
            # attention: product of above     = [0, 0, 0, 1, 0, 0, 0, 0]
        return attention


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
      
      # self.attention = MonotonicAttention(input_size, input_size, att_dim, init_r=-4)
      self.attention = ContentBasedAttention(input_size, input_size, att_dim)
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
      # return torch.Tensor(batch_size, num_layers, decoder_dim).normal_()
      return torch.Tensor(batch_size, num_layers, decoder_dim).zero_()
    
    

    # Training forward
    # def forward(self, x: torch.Tensor, y: torch.tensor) -> torch.Tensor:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      # CNN encoder
      # print(f'Input shape: {x.shape}')
      x = self.cnn_encoder(x)
      # print(f'CNN output shape: {x.shape}')
      x = x.flatten(1, 2).permute(0, 2, 1)
      # print(f'Flattened CNN output shape: {x.shape}')
      
      # RNN encoder
      self.rnn_encoder.flatten_parameters()
      x, _ = self.rnn_encoder(x)

      # Return x for CTC loss
      encoder_outputs = x.clone()
      cnn_output = self.ctc_pred(x)
      
      return cnn_output 

      # Decoder 
      batch_size, enc_sequence_length, enc_dim = x.size()
      batch_size, dec_sequence_length = y.size()

      output = self.init_x(batch_size).to(x.device)
      h = self.init_h(batch_size, enc_dim, self.layers_decoder).to(x.device).permute(1, 0, 2).contiguous()
      c_0 = torch.zeros(batch_size, self.layers_decoder, enc_dim).to(x.device).permute(1, 0, 2).contiguous()
      context = c_0
      alpha = None
      logit_list = []
      # append B, 1 with zeros to logit_list
      logit_list.append(torch.ones(batch_size, self.vocab_size).to(x.device) * self.tokenizer.bos_id)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      # breakpoint()
      
      for i in range(dec_sequence_length):
          output = self.char_embedding(output)
          x, (h, context) = self.decoder(output, (h, context))
          
          # Attention 
          alpha = self.attention.soft(encoder_outputs, h, alpha)

          # Weighted-sum
          # [batch_size, out_dim]
          context = torch.sum(alpha.unsqueeze(-1) * encoder_outputs, dim=1)
          context = context.unsqueeze(0)

          # [batch_size, out_dim]
          attentional = self.act(self.combine_c_h(torch.cat([context, h], dim=-1))).reshape(batch_size, -1)
          # print(f'attentional: {attentional.shape}')

          # [batch_size, vocab_size]
          logit = self.output(attentional)
          # print(f'Logit shape: {logit.shape}')
          logit_list.append(logit)

          output = y[:, i] # Teacher forcing
          output = output.unsqueeze(1)
          # print(f'Output shape in teacher forcing: {output.shape}')
          
      # print(f'torch.stack(logit_list, dim=1): {torch.stack(logit_list, dim=1).shape}')

      return cnn_output.log_softmax(-1).permute(1, 0, 2), torch.stack(logit_list, dim=1)


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
      monotonic_att = None
      output_list = []
      output_list.append(torch.ones(batch_size).to(x.device) * self.tokenizer.bos_id)
      
      # Flatten parameters
      self.decoder.flatten_parameters()
      
      for i in range(120):
          output = self.char_embedding(output)
          x, (h, context) = self.decoder(output, (h, context))

          # monotonic_att (one-hot): [batch_size, sequence_length]
          # chunkwise_attention (nonzero in chunk size): [batch_size, sequence_length]
          # Attention
          monotonic_att = self.attention.hard(encoder_outputs, h, monotonic_att)

          # Weighted-sum
          # [batch_size, out_dim]
          context = torch.sum(monotonic_att.unsqueeze(-1) * encoder_outputs, dim=1)
          context = context.unsqueeze(0)

          # [batch_size, out_dim]
          # attentional = self.act(self.combine_c_h(torch.cat([context, h], dim=1))).reshape(batch_size, -1)
          attentional = self.act(self.combine_c_h(torch.cat([context, h], dim=-1))).reshape(batch_size, -1)

          # [batch_size, vocab_size]
          logit = self.output(attentional)

          # Greedy Decoding
          logit = torch.argmax(logit, dim=-1)
          output = logit.unsqueeze(1)
          # print(f'Output shape in greedy decoding: {output.shape}')
          output_list.append(logit)

      # print(f'VAL torch.stack(output_list, dim=1): {torch.stack(output_list, dim=1).shape}')
      return torch.stack(output_list, dim=1)

if __name__ == "__main__":
    _ = CRNN_Michael()




    


      

