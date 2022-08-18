import torch
import torch.nn as nn
from layers import Attention,MLP

class Block(nn.Module):
    """
    Parameters
    d_model: embedding dimension
    n_heads: number of attention heads
    mlp_ratio: combined with 'd_model', determines hidden dimension of MLP layer
    qkv_bias: if True, include bias to the query, key, value projections
    """
    """
    Attributes
    norm1, norm2: LayerNorm
    attn_layer: Attention module
    mlp_layer: MLP module
    """
    def __init__(self, d_model, n_heads, mlp_ratio=4.0, qkv_bias=True, attn_p=0., p=0.):
        super().__init__()
        self.attn_layer = Attention(d_model=d_model, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=p)
        self.d_hidden = int(d_model * mlp_ratio)
        self.mlp_layer = MLP(in_features=d_model, hidden_features=self.d_hidden, out_features=d_model, p=p)
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)    ## eps: 1e-6 -> because of pretrained model
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self,x):
        """
        input shape    : [batch_size, n_patches+1, d_model]
        output shape   : [batch_size, n_patches+1, d_model]
        """
        x = x + self.norm1(self.attn_layer(x))
        x = x + self.norm2(self.mlp_layer(x))

        return x