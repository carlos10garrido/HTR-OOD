
import torch
from torch import nn
from PIL import Image
import math
from src.data.components.tokenizers import Tokenizer

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1).float()
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)

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

class TransformerKangTorch(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        image_size: tuple = (32, 100),
        use_backbone: bool = True,
        patch_per_column: bool = True,
        patch_size: int = 4,
        vocab_size: int = 0,
        max_position_embeddings: int = 1024,
        encoder_layers: int = 4,
        encoder_ffn_dim: int = 1024,
        encoder_attention_heads: int = 8,
        decoder_layers: int = 4,
        decoder_ffn_dim: int = 1024,
        decoder_attention_heads: int = 8,
        activation_function: int = "relu",
        d_model: int = 1024,
        dropout: float = 0.1,
        tokenizer: Tokenizer = None,
    ) -> None:

        super().__init__()

        self.use_backbone = use_backbone
        self.patch_per_column = patch_per_column
        self.image_size = image_size
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

        if self.use_backbone:
            self.num_channels = 512 * (image_size[0] // 16) if self.patch_per_column else 512
            self.image_size = (1, image_size[1] // 16) if self.patch_per_column else (image_size[0] // 16, image_size[1] // 16)

        else:
            self.num_channels = 3 * image_size[0] if self.patch_per_column else 3
            self.image_size = (1, image_size[1]) if self.patch_per_column else image_size
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')
        print(f'patch_per_column: {self.patch_per_column}')


        if self.use_backbone:
          self.backbone = torch.hub.load('pytorch/vision', 'resnet34', pretrained=False)

          # Unmodify 
          # Change stride of last layer (2,2) to (1,1)

          # Resnet 50 modification
          # self.backbone.layer4[0].downsample[0].stride = (1,1)
          # self.backbone.layer4[0].conv2.stride = (1,1)

          # Resnet 34 modification
          self.backbone.layer4[0].conv1.stride=(1,1)
          self.backbone.layer4[0].downsample[0].stride=(1,1)

        
          self.backbone = torch.nn.Sequential(
              *list(self.backbone.children())[:-2], # Remove last 4 layers
              # torch.nn.Conv2d(128, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          )

          # self.num_channels = 2048

        self.lin_proj = nn.Linear(self.num_channels, d_model)

        self.vocab_size = self.vocab_size

        self.pos_encoding_enc = PositionalEncoding(d_model, dropout)
        self.pos_encoding_dec = PositionalEncoding(d_model, dropout)
        self.text_embedding = nn.Embedding(self.vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=encoder_attention_heads,
            num_encoder_layers=encoder_layers,
            num_decoder_layers=decoder_layers,
            dim_feedforward=encoder_ffn_dim,
            dropout=dropout,
            activation=activation_function,
            batch_first=True
        )

        self.class_head = nn.Linear(d_model, self.vocab_size)

      
    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param images: The input tensor.
        :param labels: The labels tensor.
        :return: A tensor of predictions.
        """

        x = images
        if self.use_backbone:
          x = self.backbone(x)

        if self.patch_per_column:
          x = x.flatten(1, 2) # Flatten columns to use each flattened column as a patch
          x = x.unsqueeze(2) # unflatten to 3D
        
        x = x.permute(0, 3, 1, 2).squeeze(-1)

        x = self.lin_proj(x)
        src = self.pos_encoding_enc(x)

        # Change inputs_ids type to match labels type
        input_ids = labels.clone().type_as(labels)

        
        # Add bos to input_ids
        input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.int).to(self.device) * self.tokenizer.bos_id, input_ids], dim=-1)

        tgt = self.pos_encoding_dec(self.text_embedding(input_ids))

        tgt_key_padding_mask = input_ids == self.tokenizer.pad_id
        tgt_mask = self.transformer.generate_square_subsequent_mask(input_ids.shape[1]).to(self.device)
        

        print(f'src shape: {src.shape}. tgt shape: {tgt.shape}. tgt_mask shape: {tgt_mask.shape}. tgt_key_padding_mask shape: {tgt_key_padding_mask.shape}')
        print(f'tgt_key_padding_mask: {tgt_key_padding_mask}')

        print(f'INSIDE FORWARD')
        print(f'input_ids: {input_ids.shape}')
        print(f'input_ids[:4]: {input_ids[:4]}')

        print(f'labels: {labels.shape}')
        print(f'labels[:4]: {labels[:4]}')

        outputs = self.transformer(
            src=src,
            tgt=tgt, 
            # tgt_key_padding_mask=tgt_key_padding_mask,
            tgt_mask=tgt_mask,
            tgt_is_causal=True
        )

        outputs = self.class_head(outputs)

        return outputs
        
    
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        if self.use_backbone:
          x = self.backbone(x)

        if self.patch_per_column:
          x = x.flatten(1, 2) # unflatten to 3D
          x = x.unsqueeze(2)  

        x = x.permute(0, 3, 1, 2).squeeze(-1)

        x = self.lin_proj(x)
        src = self.pos_encoding_enc(x)
        
        # Predict manually
        preds = torch.ones((x.shape[0], 1), dtype=torch.int).to(self.device) * self.tokenizer.bos_id

        # breakpoint()

        # Prediction
        for i in range(150):
            tgt = self.pos_encoding_dec(self.text_embedding(preds))
            
            outputs = self.transformer(
                src=src,
                tgt=tgt, 
            )

            outputs = self.class_head(outputs)
            next_token = outputs[:, -1].argmax(dim=-1).unsqueeze(-1)
            preds = torch.cat([preds, next_token], dim=-1)

        return preds


if __name__ == "__main__":
    _ = TransformerKangTorch()