import torch
import torch.nn as nn


class Attention(nn.Module):
    """
    Parameters
    d_model : input and out dimension of per token features
    n_heads : number of attention heads
    qkv_bias : If True, include bias to query, key and value projections
    attn_p : dropout probability applied to the query, key and value tensors
    proj_p : dropout probability applied to the output tensor
    """
    """
    Attributes
    scale : normalizing constant for the dot product
    qkv : nn.Linear, linear projection for the query, key and value
    proj: nn.Linear, linear mapping that takes in the concatenated output of all attention heads
        and maps it into a new space
    attn_drop, proj_drop : nn.Dropout
    """
    def __init__(self, d_model, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.scale = self.d_head ** -0.5

        self.qkv = nn.Linear(in_features=d_model, out_features=d_model*3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attn_p)
        self.proj = nn.Linear(in_features=d_model, out_features=d_model)
        self.proj_drop = nn.Dropout(p=proj_p)

    def forward(self,x):
        """
        input shape : [batch_size, n_patches+1, dim]
        (+1 : class token)
        output shape should be identical
        """
        batch_size, n_patches, d_model = x.shape
        if d_model != self.d_model:
            return ValueError

        qkv = self.qkv(x)                   ## [batch_size, n_patches+1, d_model*3]
        qkv = qkv.reshape(
            batch_size, n_patches, 3, self.n_heads, self.d_head
        )                                   ## dim*3 ==split==> 3 x n_heads x head_dim  (d_model == n_heads * d_head)
                                            ## [batch_size, n_patches+1, 3, n_heads, d_head]

        qkv = qkv.permute(
            2,0,3,1,4
        )                                   ## [3, batch_size, n_heads, n_patches+1, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]    ## [batch_size, n_heads, n_patches+1, d_head]
        k_t = k.transpose(-2,-1)            ## [batch_size, n_heads, d_head, n_patches+1]
        dot_product = (q@k_t) * self.scale  ## [batch_size, n_heads, n_patches+1, n_patches+1]
        attn = dot_product.softmax(dim=-1)  ## [batch_size, n_heads, n_patches+1, n_patches+1]
        attn = self.attn_drop(attn)
        weighted_sum = attn @ v             ## [batch_size, n_heads, n_patches+1, d_head]
        weighted_sum = weighted_sum.transpose(1,2) ## [batch_size, n_patches+1, n_heads, d_head]
        weighted_sum = weighted_sum.flatten(2)     ## [batch_size, n_patches+1, d_model]

        x = self.proj(weighted_sum)
        x = self.proj_drop(x)

        return x


class MLP(nn.Module):
    """
    Parameters
    in_features : number of input features
    hidden_features : number of nodes in the hidden layer
    out_features : number of output features
    p : dropout probability
    """
    """
    Attributes
    fc1 : nn.Linear
    activation : nn.GELU
    fc2 : nn.Linear
    drop : nn.Dropout
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(p)

    def forward(self,x):
        """
        input shape     : [batch_size, n_patches+1, in_features]
        output shape    : [batch_size, n_patches+1, in_features]
        """
        x = self.drop(self.activation(self.fc1(x)))
        x = self.drop(self.fc2(x))

        return x