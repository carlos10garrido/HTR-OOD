
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


class TransformerSeg(nn.Module):
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
        activation_seg: str = 'sigmoid',
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

        if self.use_backbone:
            self.num_channels = d_model * (image_size[0] // 4) if self.patch_per_column else d_model
            self._image_size = (1, image_size[1] // 4) if self.patch_per_column else (image_size[0] // 4, image_size[1] // 4)

        else:
            self.num_channels = 1 * image_size[0] if self.patch_per_column else 3
            self._image_size = (1, image_size[1]) if self.patch_per_column else image_size
        
        print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self._image_size}')
        print(f'patch_per_column: {self.patch_per_column}')
        print(f'masking_noise: {self.masking_noise}')

        self.vit_config = ViTConfig(#SwinConfig(
            hidden_size=d_model,
            intermediate_size=encoder_ffn_dim,
            hidden_dropout_prob=dropout,
            image_size=self._image_size,
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
          
          # Make a simple CNN with 2 conv layers
          self.backbone = torch.nn.Sequential(
              torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              torch.nn.AvgPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(64),
              torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              torch.nn.AvgPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(128),
              torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
              torch.nn.ReLU(inplace=True),
              # torch.nn.MaxPool2d(kernel_size=2, stride=2), # Reduce image size by half
              torch.nn.InstanceNorm2d(256),
              torch.nn.Conv2d(256, d_model, kernel_size=1, stride=1, padding=0, bias=False),
          )

        self.decoder_config = TrOCRConfig(
            vocab_size=64*64,#(image_size[0])//2*(image_size[1])//2*3,
            # vocab_size=vocab_size,
            d_model=d_model,
            hidden_size=d_model,
            decoder_layers=decoder_layers,
            cross_attention_hidden_size=decoder_ffn_dim,
            decoder_attention_heads=decoder_attention_heads,
            decoder_ffn_dim=decoder_ffn_dim,
            scale_embedding=scale_embedding,
            use_learned_position_embeddings=True,
            activation_function=activation_function,
            output_attentions=True,
            return_dict=True,
            is_decoder=True,
            is_cross_attention=True,
            output_hidden_states=True,
        )

        self.dim_pred_image = 64*64 #(image_size[0]//2)*(image_size[1]//2)*3
        
      
        config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
           self.vit_config, 
           self.decoder_config
        )
        self.model = VisionEncoderDecoderModel(config=config)

        # Change decoder embedding layer to project images to d_model
        self.model.decoder.model.decoder.embed_tokens = torch.nn.Linear(self.dim_pred_image, d_model, bias=False)
        self.model.decoder.model.input_proj = torch.nn.Linear(self.dim_pred_image, d_model, bias=False)
        self.model.decoder.output_projection = nn.Sequential(
            torch.nn.Linear(d_model, self.dim_pred_image, bias=True),
            torch.nn.ReLU(inplace=True) if activation_seg == 'relu' else torch.nn.Sigmoid()
        )

        print(f'Printing model configs... values')
        print(f'ViTConfig: {self.vit_config.__dict__}')
        print(f'TrOCRConfig: {self.decoder_config.__dict__}')
        # print(f'TrOCRForCausalLM: {self.decoder.__dict__}')
        print(f'VisionEncoderDecoderModel: {self.model.__dict__}')
        
        self.model.config.decoder_start_token_id = 0
        self.model.config.decoder.pad_token_id = 1
        self.model.config.decoder.bos_token_id = 0
        self.model.config.decoder.eos_token_id = 2
        self.model.config.vocab_size = self.dim_pred_image
        self.model.config.pad_token_id = 1
        self.model.config.bos_token_id = 0
        self.model.config.eos_token_id = 2


    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param images: The input tensor.
        :param labels: The labels tensor.
        :return: A tensor of predictions.
        """

        x, masks = images, labels
        # print(x.shape)
        # print(masks.shape)

        if self.use_backbone:
          x = self.backbone(x)
          # print(f'x.shape: {x.shape} after conv')

        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2)
          # unflatten to 3D
          x = x.unsqueeze(2)
          # print(f'x.shape: {x.shape} after flatten')
        
        # Change inputs id 
        input_ids = labels.clone()
        # print(f'input_ids.shape: {input_ids.shape}')
        # print(f'label.shape: {labels.shape}')

        # Add bos and eos tokens
        input_ids = torch.cat(((torch.ones(input_ids.shape[0], 1, input_ids.shape[-1]) * self.model.config.bos_token_id).to(input_ids.device), input_ids), dim=1).to(input_ids.device)
        # print(f'input_ids.shape after adding bos: {input_ids.shape}')

        input_embeds = self.model.decoder.model.decoder.embed_tokens(input_ids)
        # print(f'input_ids.shape: {input_ids.shape}')

        # Predict using explictly the encoder and decoder
        output_encoder = self.model.encoder(pixel_values=x, output_attentions=True)
        # breakpoint()
        # Convert output_encoder to float
        output_encoder = output_encoder.last_hidden_state.float()
        output_decoder = self.model.decoder(inputs_embeds=input_embeds, encoder_hidden_states=output_encoder, output_attentions=True)
        # print(f'output_decoder: {output_decoder.last_hidden}')

        return output_decoder

    
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        if self.use_backbone:
          x = self.backbone(x)
          print(f'x.shape: {x.shape} after conv')

        print(f'Predict greedy x shape: {x.shape}')

        if self.patch_per_column:
          # Flatten columns to use each flattened column as a patch
          x = x.flatten(1, 2)
          # unflatten to 3D
          x = x.unsqueeze(2)
          print(f'x.shape: {x.shape} after flatten')

        input_ids = (torch.ones(x.shape[0], 1, self.dim_pred_image) * self.model.config.bos_token_id).to(x.device)
        # input_embeds = self.model.decoder.model.decoder.embed_tokens(input_ids)

        # Predict using explictly the encoder and decoder auto-regressively 
        # (we don't have stop criteria yet)
        for i in range(25):
          output_encoder = self.model.encoder(pixel_values=x, output_attentions=True)
          output_encoder = output_encoder.last_hidden_state.float()
          input_embeds = self.model.decoder.model.decoder.embed_tokens(input_ids)
          output_decoder = self.model.decoder(inputs_embeds=input_embeds, encoder_hidden_states=output_encoder, output_attentions=True)
          # print(f'output_decoder logits shape: {output_decoder.logits.shape}')
          # Cat the last prediction to the input_ids
          # pred_to_input = self.model.decoder.
          input_ids = torch.cat((input_ids, output_decoder.logits[:, -1:, :]), dim=1)
          # print(f'input_embeds.shape: {input_ids.shape}')

        print(f'OUTPUT DECODER SHAPE: {output_decoder.logits.shape}')

        return output_decoder



if __name__ == "__main__":
    _ = TransformerSeg()