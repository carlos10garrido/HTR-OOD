# %% [markdown]
# # Comparing architectures in HuggingFace (BART, BERT, Pre/Post Norm. Layers, etc.)

# %%
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    BartModel,
    BertModel,
    BertConfig,
    GenerationConfig,
    EncoderDecoderModel,
    EncoderDecoderConfig,
    RobertaPreLayerNormConfig,
)

import torch
import torch.nn as nn



# %% [markdown]
# ## BART

# %%
bart_config = BartConfig(
    vocab_size=50265,
    max_position_embeddings=1024,
    encoder_layers=12,
    encoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_layers=12,
    decoder_ffn_dim=4096,
    decoder_attention_heads=16,
    encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
    activation_function="gelu",
    d_model=1024,
    dropout=0.1,
    attention_dropout=0.0,
    activation_dropout=0.0,
    init_std=0.02,
    classifier_dropout=0.0,
    scale_embedding=False,
    use_cache=True,
    num_labels=3,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    is_encoder_decoder=True,
    decoder_start_token_id=2,
    forced_eos_token_id=2,
)

# %%
model = BartModel(bart_config)
print(model)

# %% [markdown]
# ## BERT

# %%
bert_config = BertConfig(
    vocab_size=50265,
    max_position_embeddings=1024,
    encoder_layers=12,
    encoder_ffn_dim=4096,
    encoder_attention_heads=16,
    decoder_layers=12,
    decoder_ffn_dim=4096,
    decoder_attention_heads=16,
    encoder_layerdrop=0.0,
    decoder_layerdrop=0.0,
    activation_function="gelu",
    d_model=1024,
    dropout=0.1,
    attention_dropout=0.0,
    activation_dropout=0.0,
    init_std=0.02,
    classifier_dropout=0.0,
    scale_embedding=False,
    use_cache=True,
    num_labels=3,
    pad_token_id=1,
    bos_token_id=0,
    eos_token_id=2,
    is_encoder_decoder=True,
    decoder_start_token_id=2,
    forced_eos_token_id=2,
)

# %%
model = BertModel(bert_config)
print(model)

# %% [markdown]
# ## Conclusions:
# - BART model uses Post-normalization layers as in the original "Attention is All You Need"
# - BERT model uses Pre-normalization layers thorugh Embeddding Normalization

# %%


# %%


# %% [markdown]
# ## Custom PositionalEncoding

# %%
import math

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=256):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         print(f'PE: {pe.shape}')
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         print(f'X shape in forward: {x.shape}')
#         print(f'PE shape in forward: {self.pe.shape}')
#         x = x + self.pe[:x.size(1)]
#         return self.dropout(x)

import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1).squeeze(1)
        print(f'PE: {pe.shape}')
        self.register_buffer('pe', pe)

    def forward(self, x):
        print(f'X shape in forward: {x.shape}')
        print(f'PE shape in forward: {self.pe.shape}')
        x = x + self.pe[:x.size(1)]
        print(f'Returning x as {x.shape} in PE')
        return self.dropout(x).unsqueeze(-2)

# %%
# pe_encoder, pe_decoder = PositionalEncoding(256), PositionalEncoding(256)

# model_config = BertConfig(
#     vocab_size=100,
#     d_model=256,
#     encoder_layers=4,
#     decoder_layers=4,
#     encoder_attention_heads=4,
#     decoder_attention_heads=4,
#     encoder_ffn_dim=1024,
#     decoder_ffn_dim=1024,
#     max_position_embeddings=512, # Will be deleted
#     activation_function='relu',
#     encoder_layerdrop=0.0,
#     decoder_layerdrop=0.0,
#     pad_token_id=2,
#     scale_embedding=True,
#     attention_dropout=0.0,
#     attention_probs_dropout_prob = 0.0,
#     type_vocab_size = 2,
#     init_std = 0.02,
#     initializer_range = 0.02,
#     layer_norm_eps = 1e-12,
#     activation_dropout=0.0,
#     is_encoder_decoder=True,
#     force_bos_token_to_be_generated=True,
#     use_cache=False,
#     dropout=0.1,
# )

# # model = BartForConditionalGeneration(model_config)
# # print(model)
# self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.encoder_config, self.decoder_config)
# self.model = EncoderDecoderModel(config=self.config)

# # Change positional encoding to the transformer
# model.model.encoder.embed_positions = pe_encoder
# model.model.decoder.embed_positions = pe_decoder
# # model.model.shared = nn.Identity()
# # model.model.shared = nn.Identity()

# # %%
# print(model)

# # %%
# print(model.model.encoder)

# # %%
# print(model.model.encoder.embed_positions)

# # %%
# rand_batch = torch.rand((16, 128, 1, 256)) # Sequence of 128 columns with 256 channels
# breakpoint()
# pos_enc_batch = model.model.encoder.embed_positions(rand_batch)
# print(f'Pos enc batch: {pos_enc_batch.shape}')
# breakpoint()
# enc_images = model.model.encoder(inputs_embeds=rand_batch)

# # %%
# import matplotlib.pyplot as plt
# P = model.model.encoder.embed_positions.pe.squeeze(1)
# cax = plt.matshow(P)
# plt.gcf().colorbar(cax)

# # %%

############################################################################################################

encoder_config = BertConfig(
          vocab_size = 100,
          hidden_size = 256,
          num_hidden_layers = 4,
          num_attention_heads = 4,
          intermediate_size =  1024,
          hidden_act = 'gelu',
          hidden_dropout_prob = 0.1,
          attention_probs_dropout_prob = 0.0,
          max_position_embeddings = 512,
          type_vocab_size = 2,
          initializer_range = 0.02,
          layer_norm_eps = 1e-12,
          pad_token_id = 2,
          bos_token_id = 0,
          eos_token_id = 1,
          position_embedding_type = 'absolute',
          use_cache = True,
          classifier_dropout =0.1,
          is_decoder=False  
)

decoder_config = BertConfig(
  vocab_size = 100,
  hidden_size = 256,
  num_hidden_layers = 4,
  num_attention_heads = 4,
  intermediate_size =  1024,
  hidden_act = 'gelu',
  hidden_dropout_prob = 0.1,
  attention_probs_dropout_prob = 0.0,
  max_position_embeddings = 512,
  type_vocab_size = 2,
  initializer_range = 0.02,
  layer_norm_eps = 1e-12,
  pad_token_id = 2,
  bos_token_id = 0,
  eos_token_id = 1,
  position_embedding_type = 'absolute',
  use_cache = True,
  classifier_dropout =0.1,
  is_decoder=True
)

# breakpoint()
config = EncoderDecoderConfig.from_encoder_decoder_configs(encoder_config, decoder_config)
model = EncoderDecoderModel(config=config)
        
print(model)

rand_batch = torch.rand((16, 128, 256)) # Sequence of 128 columns with 256 channels
# breakpoint()
# pos_enc_batch = model.encoder.embed_positions(rand_batch)
# print(f'Pos enc batch: {pos_enc_batch.shape}')
enc_images = model(
  inputs_embeds=rand_batch,
  decoder_input_ids=torch.randint(0, 100, (16, 128)),
)
breakpoint()
print(f'Enc images: {enc_images.keys()}')






