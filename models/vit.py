import torch
import torch.nn as nn
import math
from models.layers import ViTBlock,_make_norm

class PatchEmbed(nn.Module):
    def __init__(self,img_size=32,patch_size=4,in_chan=3,d=192):
        super(PatchEmbed,self).__init__()

        assert img_size%patch_size==0
        self.img_size=img_size
        self.patch_size=patch_size
        self.grid_size=img_size//patch_size
        self.n_patches=self.grid_size**2
        self.proj=nn.Conv2d(in_chan,d,kernel_size=patch_size,stride=patch_size)

        nn.init.trunc_normal_(self.proj.weight,std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self,x):
        B,C,H,W=x.shape
        assert H==W==self.img_size
        h=self.proj(x)
        h=h.flatten(2).permute(0,2,1).contiguous()
        return h


class VIT(nn.Module):
    def __init__(self,img_size=32,patch_size=4,in_chan=3,d_model=192,num_layers=6,num_heads=3,norm_type='layer_norm',
                 ffn_type='gelu_mlp'):
        super(VIT,self).__init__()
        resid_std=0.02/math.sqrt(2*num_layers)
        self.grid_size=img_size//patch_size
        self.n_patches=self.grid_size**2
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_chan=in_chan,d=d_model)
        self.pos_embed=nn.Parameter(torch.zeros(1,self.n_patches,d_model))
        self.blocks=nn.ModuleList([
            ViTBlock(d_model=d_model,num_heads=num_heads,resid_std=resid_std,norm_type=norm_type,ffn_type=ffn_type)
            for _ in range(num_layers)
        ])
        self.norm=_make_norm(norm_type,d_model)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)

    def forward(self,x,idx=None):
        if idx is None:
            idx=torch.arange(self.n_patches,device=x.device)
        tokens=self.patch_embed(x)[:,idx,:]
        tokens=tokens+self.pos_embed[:,idx,:]
        for block in self.blocks:
            tokens=block(tokens)
        return self.norm(tokens)

    def forward_pooled(self,x,idx=None):
        return self.forward(x,idx=idx).mean(dim=1)

if __name__ == "__main__":
    v = VIT()
    x = torch.randn(2, 3, 32, 32)

    print(v(x).shape)                                  # (2, 64, 192)
    print(v.forward_pooled(x).shape)                    # (2, 192)
    print(v(x, idx=torch.arange(0, 64, 2)).shape)       # (2, 32, 192)
    print(sum(p.numel() for p in v.parameters()))       # 2691264

    v(x).sum().backward()
    print(v.pos_embed.grad is not None, v.pos_embed.grad.abs().sum().item() > 0)
    print(v.patch_embed.proj.weight.grad is not None)