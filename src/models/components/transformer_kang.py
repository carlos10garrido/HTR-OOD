
import torch
from torch import nn
from PIL import Image
import math

# Import BART model from transformers
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    BartModel,
)

from src.data.components.tokenizers import Tokenizer

class TransformerKang(nn.Module):
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
        encoder_layerdrop: float = 0.0,
        decoder_layerdrop: float = 0.0,
        activation_function: int = "relu",
        d_model: int = 1024,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        init_std: float = 0.02,
        scale_embedding: bool = False,
        use_cache: bool = True,
        max_length: int = 100,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        is_encoder_decoder: bool = True,
        decoder_start_token_id: int = 0,
        forced_eos_token_id: int = 2,
        tokenizer: Tokenizer = None,
    ) -> None:

        super().__init__()

        self.use_backbone = use_backbone
        self.patch_per_column = patch_per_column
        self.image_size = image_size
        self.vocab_size = tokenizer.vocab_size
        self.tokenizer = tokenizer

        if self.use_backbone:
            self.num_channels = 2048 * (image_size[0] // 16) if self.patch_per_column else 2048
            self.image_size = (1, image_size[1] // 16) if self.patch_per_column else (image_size[0] // 16, image_size[1] // 16)

        else:
            self.num_channels = 3 * image_size[0] if self.patch_per_column else 3
            self.image_size = (1, image_size[1]) if self.patch_per_column else image_size
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')
        print(f'patch_per_column: {self.patch_per_column}')


        if self.use_backbone:
          self.backbone = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)

          # Unmodify 
          # Change stride of last layer (2,2) to (1,1)
          self.backbone.layer4[0].downsample[0].stride = (1,1)
          self.backbone.layer4[0].conv2.stride = (1,1)


          self.backbone = torch.nn.Sequential(
              *list(self.backbone.children())[:-2], # Remove last 4 layers
              # torch.nn.Conv2d(128, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          )

          # self.num_channels = 2048

        self.lin_proj = nn.Linear(self.num_channels, d_model)

        self.config = BartConfig(
            hidden_size=d_model,
            vocab_size=self.vocab_size,
            encoder_ffn_dim=encoder_ffn_dim,
            decoder_ffn_dim=decoder_ffn_dim,
            hidden_dropout_prob=dropout,
            image_size=self.image_size,
            patch_size=patch_size,
            d_model=d_model,
            encoder_layers=encoder_layers,
            decoder_layers=decoder_layers,
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            decoder_attention_heads=decoder_attention_heads,
            activation_function=activation_function,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            decoder_start_token_id=self.tokenizer.bos_id,
            init_std=init_std,
            decoder_layerdrop=decoder_layerdrop,
            use_cache=use_cache,
            scale_embedding=scale_embedding,
            pad_token_id=self.tokenizer.pad_id,
            bos_token_id=self.tokenizer.bos_id,
            eos_token_id=self.tokenizer.eos_id
        )

        print(f'CONFIG: {self.config}')
        
        self.config.max_length = max_length
        self.model = BartForConditionalGeneration(self.config)
        self.model.config.vocab_size = self.vocab_size
        self.model.config.max_length = 150
        self.model.config.pad_token_id = self.tokenizer.pad_id
        self.model.config.bos_token_id = self.tokenizer.bos_id
        self.model.config.eos_token_id = self.tokenizer.eos_id
        self.model.config.forced_eos_token_id = self.tokenizer.eos_id
        self.model.config.decoder_start_token_id = self.tokenizer.bos_id

        print(f'MODEL: {self.model}')

        # # Delete bart encoder shared and embed tokens
        # del self.model.get_encoder().shared
        # del self.model.get_encoder().embed_tokens

      


    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param images: The input tensor.
        :param labels: The labels tensor.
        :return: A tensor of predictions.
        """

        x = images
        # print(f'x.shape: {x.shape} before conv')
        if self.use_backbone:
          x = self.backbone(x)
          # print(f'x.shape: {x.shape} after conv')


        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2)
          # unflatten to 3D
          x = x.unsqueeze(2)
          # print(f'x.shape: {x.shape} after flatten')
        
        x = x.permute(0, 3, 1, 2).squeeze(-1)
        # print(f'x.shape: {x.shape} after permute')

        x = self.lin_proj(x)
        # print(f'x.shape: {x.shape} after lin_proj')
        

        input_ids = labels.clone()

        # Change inputs_ids type to match labels type
        input_ids = input_ids.type_as(labels)


        # Change -100 tokens to pad tokens
        # print(input_ids[input_ids == -100])
        input_ids[input_ids == -100] = self.model.config.pad_token_id
        # Add bos to input_ids
        input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.int).to(self.model.device) * self.model.config.bos_token_id, input_ids], dim=-1)

          
        outputs = self.model(inputs_embeds=x, decoder_input_ids=input_ids[:, :-1], # -1 for ignoring the last token
                          labels=labels, output_attentions=True, output_hidden_states=True)

        return outputs
        
    
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        # print(f'GREEDY PREDICT')
        if self.use_backbone:
          x = self.backbone(x)
          # print(f'x.shape: {x.shape} after conv')

        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2) # unflatten to 3D
          x = x.unsqueeze(2)
          # print(f'x.shape: {x.shape} after flatten')


        x = x.permute(0, 3, 1, 2).squeeze(-1)
        # print(f'x.shape: {x.shape} after permute')

        x = self.lin_proj(x)
        # print(f'x.shape: {x.shape} after lin_proj')

        
        return self.model.generate(
          inputs_embeds=x,
          return_dict_in_generate=True,
          output_attentions=False,
          output_hidden_states=False,
          max_length=150,
          decoder_start_token_id=self.tokenizer.bos_id,
          eos_token_id=self.tokenizer.eos_id,
          pad_token_id=self.tokenizer.pad_id,
          forced_eos_token_id=self.tokenizer.eos_id
        )


if __name__ == "__main__":
    _ = TransformerOCR()