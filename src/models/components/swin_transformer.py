
import torch
from torch import nn
from PIL import Image
from transformers import (
    TrOCRConfig,
    Swinv2Config,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from typing import List


class SwinTransformer(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        # Tuple of ints for image size
        image_size: tuple = (32, 100),
        patch_size: int = 4,
        use_backbone: bool = False,
        patch_per_column: bool = False,
        num_channels: int = 3,
        vocab_size: int = 101,
        d_model: int = 1024,
        encoder_layers: List[int] = [2, 2, 6, 2],
        encoder_attention_heads: List[int] = [3, 6, 12, 24],
        decoder_layers: int = 4,
        decoder_attention_heads: int = 4,
        decoder_ffn_dim: int = 1024,
        activation_function: str = 'gelu',
        masking_noise: float = 0.0,
        dropout: float = 0.1,
        window_size: int = 7,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.0,
        drop_path_rate: float = 0.1,
        hidden_act: str = "gelu",
        encoder_stride: int = 32,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> None:
        super().__init__()
        

        self.image_size = image_size
        self.masking_noise = masking_noise
        self.num_channels = num_channels
        self.use_backbone = use_backbone
        self.patch_per_column = patch_per_column
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')

        self.swin_config = Swinv2Config(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels, 
            embed_dim=d_model,
            depths=encoder_layers,
            num_heads=encoder_attention_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            encoder_stride=encoder_stride,
            use_absolute_position_embeddings=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
            # pad_token_id=pad_token_id,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id
        )


        if self.use_backbone:
          # self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
          # self.backbone = torch.nn.Sequential(
          #     *list(self.backbone.children())[:-4], # Remove last 4 layers
          #     torch.nn.Conv2d(128, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          # )
          
          # Make a simple CNN with 3 conv layers mantaining the same image size
          self.backbone = torch.nn.Sequential(
              torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.InstanceNorm2d(64),
              torch.nn.ReLU(inplace=True),
              torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.InstanceNorm2d(128),
              torch.nn.ReLU(inplace=True),
              torch.nn.Conv2d(128, d_model, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.InstanceNorm2d(d_model),
              torch.nn.ReLU(inplace=True),
          )


          # self.backbone = torch.nn.Sequential( # Only 1 without reduction
          #     torch.nn.Conv2d(3, d_model, kernel_size=3, stride=1, padding=1, bias=False),
          #     torch.nn.InstanceNorm2d(d_model),
          #     torch.nn.ReLU(inplace=True),
          # )


        self.decoder_config = TrOCRConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            activation_function=activation_function,
            output_attentions=True,
            return_dict=True,
            output_hidden_states=True,
            # pad_token_id=pad_token_id,
            # bos_token_id=bos_token_id,
            # eos_token_id=eos_token_id,
        )
        # self.encoder = SwinModel(self.swin_config)
        # self.decoder = TrOCRForCausalLM(
        #     TrOCRConfig(
        #         vocab_size=vocab_size,
        #         d_model=d_model,
        #         decoder_layers=decoder_layers,
        #         decoder_attention_heads=decoder_attention_heads,
        #         decoder_ffn_dim=decoder_ffn_dim,
        #         scale_embedding=scale_embedding,
        #         activation_function=activation_function,
        #         output_attentions=True,
        #         return_dict=True,
        #         output_hidden_states=True,
        #     ),
        # )
        
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
           self.swin_config,
           self.decoder_config
            
        )
        self.model = VisionEncoderDecoderModel(config=config)


        print(f'Printing model configs... keys')
        print(f'Swinv2Config: {self.swin_config.__dict__.keys()}')

        print(f'Printing model configs... values')
        print(f'Swinv2Config: {self.swin_config.__dict__}')
        print(f'TrOCRConfig: {self.decoder_config.__dict__}')
        # print(f'VisionEncoderDecoderModel: {self.model.__dict__}')
        print(self.model)
        # breakpoint()
        
        # PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3
        # 'pad_token_id': 1, bos_token_id': 0,  'eos_token_id': 2,
        # self.model.config.encoder.output_hidden_states = True
        self.model.config.decoder_start_token_id = 0
        self.model.config.decoder.pad_token_id = 1
        self.model.config.decoder.bos_token_id = 0
        self.model.config.decoder.eos_token_id = 2
        self.model.config.vocab_size = vocab_size
        self.model.config.pad_token_id = 1
        self.model.config.bos_token_id = 0
        self.model.config.eos_token_id = 2

        # breakpoint()


        # Change model decoder vocab size
        # self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")


    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param images: The input tensor.
        :param labels: The labels tensor.
        :return: A tensor of predictions.
        """
        # print(f'FORWARD PASS')
        # print(f'x.shape: {images.shape}')

        x = images
        if self.use_backbone:
          x = self.backbone(x)
          print(f'x.shape: {x.shape} after conv')
        # else:


        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2)
          # unflatten to 3D
          x = x.unsqueeze(2)
          print(f'x.shape: {x.shape} after flatten')
        
        # Change inputs id and setting a 20% ratio (without havving into account bos and eos tokens)
        input_ids = labels.clone()

        # print(f'input_ids.shape: {input_ids.shape}')

        # Select positions to be masked where token are different from bos and eos tokens and pad tokens
        mask_positions = (input_ids != self.model.config.bos_token_id) & (input_ids != self.model.config.eos_token_id) & (input_ids != self.model.config.pad_token_id) & (input_ids != -100)
        
        # Change 20% of mask positions randomly between index 3 and vocab_size - 1 (without having into account bos and eos tokens)
        mask_positions = mask_positions.to(self.model.device)
        mask_positions = mask_positions & (torch.rand(input_ids.shape).to(self.model.device) < self.masking_noise).to(self.model.device)
        # print(f'mask_positions.shape: {mask_positions.shape}')
        # print(f'mask_positions: {mask_positions}')
        # Assign mask positions to inputs ids
        if self.masking_noise > 0.0:
          input_ids[mask_positions] = torch.randint(3, self.model.config.vocab_size - 3, input_ids[mask_positions].shape).to(self.model.device)
          # input_ids[mask_positions] = -100

        # Change inputs_ids type to match labels type
        input_ids = input_ids.type_as(labels)

        # input_ids requires gradin order to be used as labels
        # input_ids.requires_grad = True

        # print(f'input_ids.shape: {input_ids.shape}')
        # print(f'input_ids: {input_ids}')
        # print(f'labels.shape: {labels.shape}')
        # print(f'labels: {labels}')

        # Permute input_ids to match model input shape

        # Change -100 tokens to pad tokens
        input_ids[input_ids == -100] = self.model.config.decoder.pad_token_id
        # Add bos to input_ids
        input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.long).to(self.model.device) * self.model.config.decoder.bos_token_id, input_ids], dim=-1)

        # Predict using explictly the encoder and decoder
        # output_encoder = self.encoder(pixel_values=x, output_attentions=True)
        # output_decoder = self.decoder(input_ids=input_ids, encoder_hidden_states=output_encoder, output_attentions=True)
          
        # return output_decoder
        # print(f'input_ids[:5]: {input_ids[:5]}')
        # print(f'labels[:5]: {labels[:5]}')


        # breakpoint()
        return self.model(pixel_values=x, decoder_input_ids=input_ids[:, :-1],
                          labels=labels, output_attentions=True, output_hidden_states=True)

        # return self.model(pixel_values=x, labels=labels, output_attentions=True, output_hidden_states=True)
    
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        if self.use_backbone:
          x = self.backbone(x)
          print(f'x.shape: {x.shape} after conv')

        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2)
          # unflatten to 3D
          x = x.unsqueeze(2)
          print(f'x.shape: {x.shape} after flatten')

        # Predict returning attention weights
        # Predict using explictly the encoder and decoder
        # output_encoder = self.encoder(pixel_values=x, output_attentions=True)
        # output_decoder = self.decoder.generate(
        #    encoder_hidden_states=output_encoder, 
        #    return_dict_in_generate=True, 
        #    output_attentions=True, 
        #    output_hidden_states=True
        # )
          
        # return output_decoder
          
        # print(self.model.dtype)
        # print(x.dtype)
        # print(x.shape)
        

        return self.model.generate(x, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)


if __name__ == "__main__":
    _ = SwinTransformer()