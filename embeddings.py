import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size)

    def forward(self,x):
        """
        input shape : [batch_size, in_channels, img_size, img_size]
        output shape : [batch_size, n_patches, d_model]
        """

        x = self.proj(x)            ## [batch_size, d_model ,(img_size//patch_size), (img_size//patch_size)]
        x = x.flatten(2)            ## [batch_size, d_model, n_patches]
        x = x.transpose(1, 2)       ## [batch_size, n_patches, d_model]

        return x


