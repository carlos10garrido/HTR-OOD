
import torch
from torch import nn
from PIL import Image
from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
    SwinModel,
    SwinConfig,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig
)


class TransformerOCR(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        # Tuple of ints for image size
        image_size: tuple = (32, 100),
        use_backbone: bool = True,
        patch_per_column: bool = True,
        patch_size: int = 4,
        vocab_size: int = 101,
        d_model: int = 1024,
        encoder_layers: int = 8,
        encoder_attention_heads: int = 4,
        encoder_ffn_dim: int = 1024,
        decoder_layers: int = 4,
        decoder_attention_heads: int = 4,
        decoder_ffn_dim: int = 1024,
        activation_function: str = 'gelu',
        max_position_embeddings: int = 1024,
        masking_noise: float = 0.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.0,
        activation_dropout: float = 0.0,
        decoder_start_token_id: int = 0,
        init_std: float = 0.02,
        decoder_layerdrop: float = 0.0,
        use_cache: int = True,
        scale_embedding: int = True,
        use_learned_position_embeddings: int = True,
        layernorm_embedding: int = True,
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

        self.use_backbone = use_backbone
        self.patch_per_column = patch_per_column
        self.image_size = image_size
        self.masking_noise = masking_noise
        self.vocab_size = vocab_size

        if self.use_backbone:
            self.num_channels = d_model * (image_size[0] // 8) if self.patch_per_column else d_model
            self.image_size = (1, image_size[1] // 8) if self.patch_per_column else (image_size[0] // 8, image_size[1] // 8)

        else:
            self.num_channels = 3 * image_size[0] if self.patch_per_column else 3
            self.image_size = (1, image_size[1]) if self.patch_per_column else image_size
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')
        print(f'patch_per_column: {self.patch_per_column}')
        print(f'masking_noise: {self.masking_noise}')

        self.vit_config = ViTConfig(#SwinConfig(
            hidden_size=d_model,
            intermediate_size=encoder_ffn_dim,
            hidden_dropout_prob=dropout,
            image_size=self.image_size,
            patch_size=patch_size,
            d_model=d_model,
            num_channels=self.num_channels, # 128 is the number of channels from the resnet18 backbone
            num_hidden_layers=encoder_layers,
            num_attention_heads=encoder_attention_heads,
            activation_function=activation_function,
            max_position_embeddings=max_position_embeddings,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            decoder_start_token_id=decoder_start_token_id,
            init_std=init_std,
            decoder_layerdrop=decoder_layerdrop,
            use_cache=use_cache,
            scale_embedding=scale_embedding,
            use_learned_position_embeddings=use_learned_position_embeddings,
            layernorm_embedding=layernorm_embedding,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id
        )

        if self.use_backbone:
          self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
          self.backbone = torch.nn.Sequential(
              *list(self.backbone.children())[:-4], # Remove last 4 layers
              torch.nn.Conv2d(128, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          )
          
          # Make a simple CNN with 2 conv layers
          # self.backbone = torch.nn.Sequential(
          #     torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
          #     torch.nn.ReLU(inplace=True),
          #     torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
          #     torch.nn.InstanceNorm2d(64),
          #     torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
          #     torch.nn.ReLU(inplace=True),
          #     torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
          #     torch.nn.InstanceNorm2d(64),
          #     torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
          #     torch.nn.ReLU(inplace=True),
          #     # torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
          #     torch.nn.InstanceNorm2d(64),
          #     torch.nn.Conv2d(256, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          # )

          # self.backbone = torch.nn.Sequential( # Only 1 without reduction
          #     torch.nn.Conv2d(3, d_model, kernel_size=3, stride=1, padding=1, bias=False),
          #     torch.nn.InstanceNorm2d(d_model),
          #     torch.nn.ReLU(inplace=True),
          # )

        # self.encoder = ViTModel(self.vit_config)

        self.decoder_config = TrOCRConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            decoder_layers=decoder_layers,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            scale_embedding=scale_embedding,
            activation_function=activation_function,
            output_attentions=True,
            return_dict=True,
            output_hidden_states=True,
        )
       
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
           self.vit_config, 
           self.decoder_config
        )
        # self.model = VisionEncoderDecoderModel(config=config)

        
        self.model = VisionEncoderDecoderModel(config=config)

        print(f'Printing model configs... values')
        print(f'ViTConfig: {self.vit_config.__dict__}')
        print(f'TrOCRConfig: {self.decoder_config.__dict__}')
        print(f'VisionEncoderDecoderModel: {self.model.__dict__}')
        # breakpoint()
        
        # PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3
        self.model.config.decoder_start_token_id = 0
        self.model.config.decoder.pad_token_id = 1
        self.model.config.decoder.bos_token_id = 0
        self.model.config.decoder.eos_token_id = 2
        self.model.config.vocab_size = vocab_size
        self.model.config.max_length = 64
        self.model.config.pad_token_id = 1
        self.model.config.bos_token_id = 0
        self.model.config.eos_token_id = 2


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

        # Change inputs_ids type to match labels type
        input_ids = input_ids.type_as(labels)


        # Change -100 tokens to pad tokens
        input_ids[input_ids == -100] = self.model.config.decoder.pad_token_id
        # Add bos to input_ids
        input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.int).to(self.model.device) * self.model.config.decoder.bos_token_id, input_ids], dim=-1)

          
        return self.model(pixel_values=x, decoder_input_ids=input_ids[:, :-1], # -1 for ignoring the last token
                          labels=labels, output_attentions=True, output_hidden_states=True)
    
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
          x = x.flatten(1, 2) # unflatten to 3D
          x = x.unsqueeze(2)
          print(f'x.shape: {x.shape} after flatten')

        
        return self.model.generate(x, return_dict_in_generate=True, output_attentions=True, output_hidden_states=True)


if __name__ == "__main__":
    _ = TransformerOCR()