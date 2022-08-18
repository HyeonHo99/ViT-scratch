import torch
import torch.nn as nn
from embeddings import PatchEmbed
from blocks import Block


class VisionTransformer(nn.Module):
    """
    Parameters
    img_size, patch_size, in_channels, n_classes
    d_model: dimensionality of patch embeddings
    depth: number of transformer blocks
    n_heads, mlp_ratio, qkv_bias, attn_p, p
    """
    """
    Attributes
    patch_embed: PatchEmbed Module
    cls_token : nn.Parameter
        class token, the first token in the sequence
    pos_embed : nn.Parameter
        Positional embeddings of the cls token(patch) + all image patches
        shape : [n_patches+1, embed_dim]
    pos_drop : nn.Dropout
    blocks : nn.ModuleList
        List of Block Modules
    norm : nn.LayerNorm
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3,
                 n_classes=1000, d_model=512, depth=12, n_heads=12,
                 mlp_ratio=4., qkv_bias=True, attn_p=0., p=0.):

        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, d_model=d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.patch_embed.n_patches, d_model))
        self.pos_drop = nn.Dropout(p=p)
        self.blocks = nn.ModuleList(
            [
                Block(d_model=d_model, n_heads=n_heads, mlp_ratio=mlp_ratio,
                   qkv_bias=qkv_bias, p=p, attn_p=attn_p) for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(d_model, eps=1e-6)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):
        """
        input   : [batch_size, in_channels, img_size, img_size]
        output  : [batch_size, n_classes]
        """
        batch_size = x.shape[0]
        x = self.patch_embed(x)                 ## shape : [batch_size, n_patches, d_model]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_token, x), dim=1)    ## shape : [batch_size, n_patches + 1, d_model]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        class_token = x[:,0]
        x = self.head(class_token)

        ## softmax not included inside Module
        return x