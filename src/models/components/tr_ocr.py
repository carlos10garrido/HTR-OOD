
import torch
from torch import nn
from PIL import Image
from src.data.components.tokenizers import Tokenizer
from transformers import (
    TrOCRConfig,
    TrOCRProcessor,
    TrOCRForCausalLM,
    ViTConfig,
    ViTImageProcessor,
    ViTModel,
    VisionEncoderDecoderModel,
    VisionEncoderDecoderConfig
)

# I want to import TrOCRScaledWordEmbedding
# from transformers import TrOCRScaledWordEmbedding




class TransformerOCR(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        # Tuple of ints for image size
        image_size: tuple = (32, 100),
        use_backbone: bool = True,
        patch_per_column: bool = True,
        patch_size: int = 4,
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
        tokenizer: Tokenizer = None,
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
        self.vocab_size = tokenizer.vocab_size
        self.d_model = d_model
        self.tokenizer = tokenizer
        # print(f'num_channels: {self.num_channels}')
        print(f'image_size: {self.image_size}')
        print(f'patch_per_column: {self.patch_per_column}')
        print(f'masking_noise: {self.masking_noise}')

        # self.vit_config = ViTConfig(
        #     hidden_size=d_model,
        #     intermediate_size=encoder_ffn_dim,
        #     hidden_dropout_prob=dropout,
        #     image_size=self.image_size,
        #     patch_size=patch_size,
        #     d_model=d_model,
        #     num_channels=3, # 128 is the number of channels from the resnet18 backbone
        #     num_hidden_layers=encoder_layers,
        #     num_attention_heads=encoder_attention_heads,
        #     activation_function=activation_function,
        #     max_position_embeddings=max_position_embeddings,
        #     dropout=dropout,
        #     attention_dropout=attention_dropout,
        #     activation_dropout=activation_dropout,
        #     decoder_start_token_id=decoder_start_token_id,
        #     init_std=init_std,
        #     decoder_layerdrop=decoder_layerdrop,
        #     use_cache=use_cache,
        #     scale_embedding=scale_embedding,
        #     use_learned_position_embeddings=use_learned_position_embeddings,
        #     layernorm_embedding=layernorm_embedding,
        #     pad_token_id=pad_token_id,
        #     bos_token_id=bos_token_id,
        #     eos_token_id=eos_token_id
        # )

        # self.decoder_config = TrOCRConfig(
        #     vocab_size=self.vocab_size,
        #     d_model=d_model,
        #     decoder_layers=decoder_layers,
        #     decoder_attention_heads=decoder_attention_heads,
        #     decoder_ffn_dim=decoder_ffn_dim,
        #     scale_embedding=scale_embedding,
        #     activation_function=activation_function,
        #     output_attentions=True,
        #     return_dict=True,
        #     output_hidden_states=True,
            
        # )
       
        # config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(
        #    self.vit_config, 
        #    self.decoder_config
        # )
        
        # # self.model = VisionEncoderDecoderModel(
        # #   encoder=ViTModel(self.vit_config),
        # #   decoder=TrOCRForCausalLM(self.decoder_config)
        # # )
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.processor.do_reescale = False
        

        # print(f'Printing model configs... values')
        # print(f'ViTConfig: {self.vit_config.__dict__}')
        # print(f'TrOCRConfig: {self.decoder_config.__dict__}')
        # print(f'VisionEncoderDecoderModel: {self.model.__dict__}')
        
        # PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 1, 0, 2, 3
        self.model.config.decoder_start_token_id = 0
        self.model.config.decoder.pad_token_id = 1
        self.model.config.decoder.bos_token_id = 0
        self.model.config.decoder.eos_token_id = 2
        self.model.config.vocab_size = self.vocab_size
        self.model.config.max_length = 128
        self.model.config.pad_token_id = tokenizer.pad_id
        self.model.config.bos_token_id = tokenizer.bos_id
        self.model.config.eos_token_id = tokenizer.eos_id
        self.model.config.decoder_start_token_id = tokenizer.bos_id
        self.model.config.decoder.pad_token_id = tokenizer.pad_id
        self.model.config.decoder.bos_token_id = tokenizer.bos_id
        self.model.config.decoder.eos_token_id = tokenizer.eos_id
        
        # self.model.config.decoder_start_token_id = 
        self.model.decoder.decoder_start_token_id = tokenizer.bos_id
        self.model.decoder.config.decoder_start_token_id = tokenizer.bos_id
        
        self.model.decoder.config.vocab_size = self.vocab_size
        self.model.decoder.output_projection = nn.Linear(self.model.decoder.config.d_model, self.vocab_size, bias=False).to(self.model.device)
        self.model.decoder.model.decoder.embed_tokens = nn.Embedding(self.vocab_size, self.model.decoder.config.d_model).to(self.model.device)
        
        print(f'Printing model configs... values. {self.model.config.__dict__}')



    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param images: The input tensor.
        :param labels: The labels tensor.
        :return: A tensor of predictions.
        """
        # print(f'FORWARD PASS')
        # print(f'x.shape: {images.shape}')

        x = images.type(torch.uint8)
        x = self.processor(x, return_tensors="pt").pixel_values.to(self.model.device)
        
        # Change inputs id and setting a 20% ratio (without havving into account bos and eos tokens)
        input_ids = labels.clone()

        # Change inputs_ids type to match labels type
        input_ids = input_ids.type_as(labels)

        # Change -100 tokens to pad tokens
        input_ids[input_ids == -100] = self.model.config.decoder.pad_token_id
        # Add bos to input_ids
        # input_ids = torch.cat([torch.ones(input_ids.shape[0], 1, dtype=torch.int).to(self.model.device) * self.model.config.decoder.bos_token_id, input_ids], dim=-1)
        
        # breakpoint()
          
        outputs = self.model(pixel_values=x, decoder_input_ids=input_ids, # -1 for ignoring the last token
                          labels=labels, output_attentions=False, output_hidden_states=False)
        return outputs
    
    def predict_greedy(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        
        # Return both indexes and raw logits
        
        x = x.type(torch.uint8)
        x = self.processor(x, return_tensors="pt").pixel_values.to(self.model.device)
        
        outputs = self.model.generate(x, decoder_start_token_id=self.model.config.decoder.bos_token_id, eos_token_id=self.model.config.decoder.eos_token_id, max_length=120, return_dict_in_generate=True, num_beams=1, output_scores=True)
        raw_preds = outputs.scores
        # convert to tensor format the scores
        raw_preds = torch.stack(raw_preds, dim=1)
        
        return outputs.sequences, raw_preds


if __name__ == "__main__":
    _ = TransformerOCR()