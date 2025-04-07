# vit.py

# Imports pytorch and pytorch lightning
import torch
import numpy as np
import pytorch_lightning as pl
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MHA(pl.LightningModule):
  def __init__(
    self,
    dim=512,
    n_heads=8,
  ) -> None:
    super(MHA, self).__init__()

    # Define query, key, value linear projections
    self.q_linear = torch.nn.Linear(dim, dim, bias=False)
    self.k_linear = torch.nn.Linear(dim, dim, bias=False)
    self.v_linear = torch.nn.Linear(dim, dim, bias=False)

    self.n_heads = n_heads
    self.dim = dim

    # Define output linear projection
    self.out = torch.nn.Sequential(
      torch.nn.LayerNorm(dim),
      torch.nn.Linear(dim, dim),
    )


  def forward(self, q, k, v) -> torch.Tensor:
    # Forward of MHA
    # x.shape = (batch_size, num_patches + 1, dim)

    # Get batch size
    batch_size = q.shape[0]

    # Get query, key, value
    q, k, v = self.q_linear(q), self.k_linear(k), self.v_linear(v)

    # Reshape query, key, value
    q = q.reshape(batch_size, -1, self.n_heads, self.dim // self.n_heads)
    k = k.reshape(batch_size, -1, self.n_heads, self.dim // self.n_heads)
    v = v.reshape(batch_size, -1, self.n_heads, self.dim // self.n_heads)

    # Transpose query, key, value
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # Calculate attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.dim // self.n_heads)

    # Apply attention dropout and softmax
    attn = torch.nn.functional.softmax(scores, dim=-1)

    # Apply attention to value
    x = torch.matmul(attn, v)

    # Transpose x
    x = x.transpose(1, 2)

    # Reshape x to merge heads
    x = x.reshape(batch_size, -1, self.dim)

    # Apply output linear projection
    x = self.out(x)

    return x, attn


class EncoderLayer(pl.LightningModule):
  def __init__(
    self,
    dim=512,
    n_heads=8,
    mlp_dim=512,
    dropout=0.1,
  ) -> None:
    super(EncoderLayer, self).__init__()

    # Define normalization layers
    self.norm1 = torch.nn.LayerNorm(dim)
    self.norm2 = torch.nn.LayerNorm(dim)

    
    # Define multi-head attention layer
    self.mha = MHA(
        n_heads=n_heads,
        dim=dim,
    )

    # Define feedforward layer
    self.ffn = torch.nn.Sequential(
        torch.nn.Linear(dim, mlp_dim),
        torch.nn.GELU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(mlp_dim, dim),
        torch.nn.Dropout(dropout),
    )

  def forward(self, x) -> torch.Tensor:
    # Forward of EncoderLayer
    # x.shape = (batch_size, num_patches + 1, dim)
    # Normalization 
    x1 = self.norm1(x)
    # Multi-head attention
    x2, att = self.mha(x1, x1, x1)
    # Residual connection
    x3 = x + x2
    # Normalization
    x4 = self.norm2(x3)
    # Feedforward
    x5 = self.ffn(x4)
    # Residual connection
    x6 = x3 + x5

    return x6, att



class ViT(pl.LightningModule):
  def __init__(
    self,
    img_size=(128, 128),
    patch_size=4,
    dim=512,
    n_layers=8,
    n_heads=8,
    mlp_dim=512,
    dropout=0.1,
  ) -> None:
    super(ViT, self).__init__()
    self.save_hyperparameters()
    self.img_size = img_size
    self.patch_size = patch_size
    self.dim = dim
    self.n_layers = n_layers
    self.n_heads = n_heads
    self.mlp_dim = mlp_dim
    self.dropout = dropout

    self.loss_fn = torch.nn.CrossEntropyLoss()

    self.train_losses = []
    self.train_accs = []
    self.val_losses = []
    self.val_accs = []

    self.num_patches = (img_size[0] // 8) * (img_size[1] // 8) + 8# Aprox 600 patcches

    print(f'Number of patches: {self.num_patches}')

    # 8 because of resnet18 backbone 
    # 128 / 8 = 16
    # 256 / 8 = 32

    # Define resnet backbone for extracting feature maps
    self.backbone = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
    self.backbone = torch.nn.Sequential(
      *list(self.backbone.children())[:-4],
      # Redim to dim for transformer
      torch.nn.Conv2d(128, self.dim, kernel_size=1, stride=1, padding=0, bias=False),
      )

    print(f'Backbone: {self.backbone}')

    # Define patch embedding layer as a resnet18 backbone + 
    self.patch_embedding = self.backbone

    # Define learnable positional embedding
    self.pos_embedding = torch.nn.parameter.Parameter(torch.randn(1, self.num_patches, self.dim))

    # # Define class token
    # self.class_token = torch.nn.parameter.Parameter(
    #     torch.randn(1, 1, self.dim)
    # )


    # Define encoder blocks
    self.encoders = torch.nn.ModuleList([
        EncoderLayer(
            dim=self.dim,
            n_heads=self.n_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout,
        ) for _ in range(self.n_layers)
    ])

  def forward(self, x) -> torch.Tensor:
    # Forward of ViT

    # Apply patch embedding
    x = self.patch_embedding(x)

    # output shape = (batch_size, dim, num_patches + 1, 1)

    # Reshape x to flatten patches [batch_size, dim, num_patches + 1]
    x = x.flatten(2).transpose(1, 2)

    # Add class token to x
    # x = torch.cat((self.class_token.repeat(batch_size, 1, 1), x), dim=1)

    # print(f'Pos embedding shape = {self.pos_embedding.shape}')

    # Add positional embedding to x
    x += self.pos_embedding[:, :x.shape[1]]

    # print(x.shape)
    att = None

    # Apply encoders
    for encoder in self.encoders:
      x, att = encoder(x)
    
    return x