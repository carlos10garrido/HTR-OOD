import torch
from torch import nn
from PIL import Image
from transformers import (
    TrOCRConfig,
    TrOCRForCausalLM,
    ConvNextV2Config,
    ConvNextV2Model,
    VisionEncoderDecoderConfig,
    VisionEncoderDecoderModel,
)

from typing import List


class ConvNext(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        # Tuple of ints for image size
        image_size: tuple = (32, 100),
        patch_size: int = 4,
        use_backbone: bool = False,
        patch_per_column: bool = False,
        num_channels: int = 3,
        hidden_sizes: List[int] = [96, 192, 384, 768],
        num_stages: int = 4,
        depths: List[int] = [3, 3, 9, 3],
        hidden_act: str = "gelu",
        vocab_size: int = 101,
        masking_noise: float = 0.0,
        drop_path_rate: float = 0.0,
        d_model: int = 512,
        dropout: float = 0.1,
        decoder_layers: int = 8,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 512,
        activation_function: str = "gelu",
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
    ) -> None:
        """Initialize a `SimpleDenseNet` module.

        :param input_size: The number of input features.
        :param lin1_size: The number of output features of the first linear layer.
        :param lin2_size: The number of output features of the second linear layer.
        :param lin3_size: The number of output features of the third linear layer.
        :param output_size: The number of output features of the final linear layer.
        """
        super().__init__()
        

        self.image_size = image_size
        self.masking_noise = masking_noise
        self.num_channels = num_channels
        self.use_backbone = use_backbone
        self.patch_per_column = patch_per_column
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')

        self.convnext_config = ConvNextV2Config(
            patch_size=patch_size,
            num_channels=num_channels, 
            num_stages=num_stages,
            hidden_sizes=hidden_sizes,
            # depths=depths,
            drop_path_rate=drop_path_rate,
            hidden_act=hidden_act,
            use_absolute_position_embeddings=False,
            output_hidden_states=True,
            output_attentions=True,
            return_dict=True,
        )


        if self.use_backbone:
          # self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
          # self.backbone = torch.nn.Sequential(
          #     *list(self.backbone.children())[:-4], # Remove last 4 layers
          #     torch.nn.Conv2d(128, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          # )
          
          # Make a simple CNN with 2 conv layers
          self.backbone = torch.nn.Sequential(
              torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(64),
              torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(64),
              torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              # torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(64),
              torch.nn.Conv2d(256, d_model, kernel_size=1, stride=1, padding=0, bias=False),
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
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )

        self.encoder = ConvNextV2Model(config=self.convnext_config)
        # print()

        print(f'Encoder layers BEFORE REMOVING: self.encoder: {self.encoder}')

        # Remove last layer of encoder
        # self.encoder.encoder.stages[-1].layers = self.encoder.encoder.stages[-1].layers[:-3]
        # Change stride of downsampling layer to 1
        # self.encoder.encoder.stages[-2].downsampling_layer[-1] = torch.nn.Conv2d(hidden_sizes[-3], hidden_sizes[-2], kernel_size=(1, 1), stride=(1, 1))
        # self.encoder.encoder.stages[-1].downsampling_layer[-1] = torch.nn.Conv2d(hidden_sizes[-2], hidden_sizes[-1], kernel_size=(1, 1), stride=(1, 1))
        # self.encoder.encoder.stages[-1].layers = self.encoder.encoder.stages[-1].layers[:-1]

        print(f'Encoder layers: AFTER REMOVING: self.encoder: {self.encoder}')

        # breakpoint()

        self.decoder = TrOCRForCausalLM(config=self.decoder_config)

        print(f'Printing model configs... keys')
        print(self.encoder.config.__dict__.keys())
        print(self.decoder.config.__dict__.keys())


        
        # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        #    self.convnext_config,
        #    self.decoder_config
            
        # )
        # self.model = VisionEncoderDecoderModel(
        #    encoder=ConvNextModel(config=self.convnext_config),
        #     decoder=TrOCRForCausalLM(config=self.decoder_config)
        # )
        # self.model = VisionEncoderDecoderModel(config=config)


        print(f'Printing model configs... keys')
        print(f'ConvNext: {self.convnext_config.__dict__.keys()}')

        print(f'Printing model configs... values')
        print(f'ConvNext: {self.convnext_config.__dict__}')
        print(f'TrOCRConfig: {self.decoder_config.__dict__}')
        # print(f'VisionEncoderDecoderModel: {self.model.__dict__}')
        # print(self.model)
        # breakpoint()

        self.decoder_config.pad_token_id = 1
        self.decoder_config.bos_token_id = 0
        self.decoder_config.eos_token_id = 2
        self.decoder_config.decoder_start_token_id = 0

        self.red_encoder = nn.Sequential(
          nn.Linear(hidden_sizes[-1]*(image_size[0]//16), d_model),
          # nn.LayerNorm(d_model),
          nn.ReLU(inplace=True),
        )
            

        
        # PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3
        # 'pad_token_id': 1, bos_token_id': 0,  'eos_token_id': 2,
        # self.model.config.encoder.output_hidden_states = True
        # self.model.config.decoder_start_token_id = 0
        # self.model.config.decoder.pad_token_id = 1
        # self.model.config.decoder.bos_token_id = 0
        # self.model.config.decoder.eos_token_id = 2
        # self.model.config.vocab_size = vocab_size
        # self.model.config.pad_token_id = 1
        # self.model.config.bos_token_id = 0
        # self.model.config.eos_token_id = 2

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
        print(f'x.shape (images): {images.shape}')

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

        # Select positions to be masked where token are different from bos and eos tokens and pad tokens
        if self.masking_noise > 0.0:
          mask_positions = (input_ids != self.model.config.bos_token_id) & (input_ids != self.model.config.eos_token_id) & (input_ids != self.model.config.pad_token_id) & (input_ids != -100)
        
          # Change 20% of mask positions randomly between index 3 and vocab_size - 1 (without having into account bos and eos tokens)
          mask_positions = mask_positions.to(self.model.device)
          mask_positions = mask_positions & (torch.rand(input_ids.shape).to(self.model.device) < self.masking_noise).to(self.model.device)

        # Assign mask positions to inputs ids
        if self.masking_noise > 0.0:
          # input_ids[mask_positions] = torch.randint(3, self.model.config.vocab_size - 3, input_ids[mask_positions].shape).to(self.model.device)
          input_ids[mask_positions] = -100

        # Change inputs_ids type to match labels type
        input_ids = input_ids.type_as(labels)

        # Change -100 tokens to pad tokens
        input_ids[input_ids == -100] = self.decoder_config.pad_token_id
        # Add bos to input_ids
        input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.long).to(self.encoder.device) * self.decoder_config.bos_token_id, input_ids], dim=-1)

        # Predict using explictly the encoder and decoder
        # output_encoder = self.encoder(pixel_values=x, output_attentions=True)
        # output_decoder = self.decoder(input_ids=input_ids, encoder_hidden_states=output_encoder, output_attentions=True)
          
        # return output_decoder

        # print(f'input_ids[:10]: {input_ids[:10]}')
        # print(f'labels[:10]: {labels[:10]}')

        # Predict with encoder-decoder model
        encodings = self.encoder(pixel_values=x, output_hidden_states=True)
        # Flatten encodings
        print(f'encodings shape: {encodings.last_hidden_state.shape}')

        encodings = encodings.last_hidden_state.flatten(1, 2).unsqueeze(2).permute(0, 3, 2, 1).squeeze(2)
        encodings = self.red_encoder(encodings)
        print(f'encodings shape: {encodings.shape}')
        outputs = self.decoder(
           input_ids=input_ids[:, :-1], 
           labels=labels,
           encoder_hidden_states=encodings,
           output_attentions=True, 
           output_hidden_states=True
        )


        # Are encodding differentiable? Check if gradients are computed
        # print(f'encodings.requires_grad: {encodings.requires_grad}')
        # print(f'outputs.loss.requires_grad: {outputs.loss.requires_grad}')


        return outputs
      


        # # breakpoint()
        # return self.model(pixel_values=x, decoder_input_ids=input_ids[:, :-1],
        #                   labels=labels, output_attentions=True, output_hidden_states=True)

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
          
        # Predict with encoder-decoder model
        encodings = self.encoder(pixel_values=x, output_hidden_states=True)
        print(f'encodings shape: {encodings.last_hidden_state.shape}')
        # print(f'encodings shape: {encodings.last_hidden_state.flatten(1, 2).unsqueeze(2).shape}')
        # Flatten encodings
        encodings = encodings.last_hidden_state.flatten(1, 2).unsqueeze(2).permute(0, 3, 2, 1).squeeze(2)
        encodings = self.red_encoder(encodings)
        # print(f'encodings shape: {encodings.shape}')

      
        outputs = self.decoder.generate(
           encoder_hidden_states=encodings,
           return_dict_in_generate=True, 
           output_attentions=True, 
           output_hidden_states=True
        )

        return outputs
        

        # return self.model.generate(x, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)


if __name__ == "__main__":
    _ = ConvNext()