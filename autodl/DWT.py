from torch.autograd import Function
import torch
import torch.nn as nn
import pywt,math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
class IDWPT_Function_1D(Function):
    @staticmethod
    def forward(ctx, x, w_lo, w_hi):
        # Assuming the input x is concatenated coefficients [x_lo_lo, x_lo_hi, x_hi_lo, x_hi_hi]
        channels = x.shape[1] // 4
        B, C, L = x.shape

        # Extract sub-bands
        x_lo_lo = x[:, :channels, :]
        x_lo_hi = x[:, channels:2*channels, :]
        x_hi_lo = x[:, 2*channels:3*channels, :]
        x_hi_hi = x[:, 3*channels:, :]

        # Reconstruct second level low and high components
        x_lo = torch.nn.functional.conv_transpose1d(x_lo_lo, w_lo.expand(channels, -1, -1), stride=2, groups=channels) \
             + torch.nn.functional.conv_transpose1d(x_lo_hi, w_hi.expand(channels, -1, -1), stride=2, groups=channels)
        x_hi = torch.nn.functional.conv_transpose1d(x_hi_lo, w_lo.expand(channels, -1, -1), stride=2, groups=channels) \
             + torch.nn.functional.conv_transpose1d(x_hi_hi, w_hi.expand(channels, -1, -1), stride=2, groups=channels)

        # Reconstruct first level
        x_reconstructed = torch.nn.functional.conv_transpose1d(x_lo, w_lo.expand(channels, -1, -1), stride=2, groups=channels) \
                        + torch.nn.functional.conv_transpose1d(x_hi, w_hi.expand(channels, -1, -1), stride=2, groups=channels)

        return x_reconstructed
class IDWPT_1D(nn.Module):
    def __init__(self, wave):
        super(IDWPT_1D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi)  # No reversal for inverse
        dec_lo = torch.Tensor(w.dec_lo)

        # Register 1D filters for low and high frequency components
        self.register_buffer('w_lo', dec_lo.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hi', dec_hi.unsqueeze(0).unsqueeze(0))

        self.w_lo = self.w_lo.to(dtype=torch.float32)
        self.w_hi = self.w_hi.to(dtype=torch.float32)

    def forward(self, x):
        return IDWPT_Function_1D.apply(x, self.w_lo, self.w_hi)
class DWPT_Function_1D(Function):
    @staticmethod
    def forward(ctx, x, w_lo, w_hi):
        channels = x.shape[1]
        # First Level Decomposition
        x_lo = torch.nn.functional.conv1d(x, w_lo.expand(channels, -1, -1), stride=2, groups=channels)
        x_hi = torch.nn.functional.conv1d(x, w_hi.expand(channels, -1, -1), stride=2, groups=channels)

        # Second Level Decomposition
        x_lo_lo = torch.nn.functional.conv1d(x_lo, w_lo.expand(channels, -1, -1), stride=2, groups=channels)
        x_lo_hi = torch.nn.functional.conv1d(x_lo, w_hi.expand(channels, -1, -1), stride=2, groups=channels)
        x_hi_lo = torch.nn.functional.conv1d(x_hi, w_lo.expand(channels, -1, -1), stride=2, groups=channels)
        x_hi_hi = torch.nn.functional.conv1d(x_hi, w_hi.expand(channels, -1, -1), stride=2, groups=channels)

        # Concatenate all results
        return torch.cat([x_lo_lo, x_lo_hi, x_hi_lo, x_hi_hi], dim=1)
class DWPT_1D(nn.Module):
    def __init__(self, wave):
        super(DWPT_1D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        # Register 1D filters for low and high frequency components
        self.register_buffer('w_lo', dec_lo.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hi', dec_hi.unsqueeze(0).unsqueeze(0))

        self.w_lo = self.w_lo.to(dtype=torch.float32)
        self.w_hi = self.w_hi.to(dtype=torch.float32)

    def forward(self, x):
        return DWPT_Function_1D.apply(x, self.w_lo, self.w_hi)
class Mlp(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, L):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class WaveAttention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.sr_ratio = sr_ratio

        self.dwt = DWPT_1D(wave='haar')
        self.idwt = IDWPT_1D(wave='haar')
        self.reduce = nn.Sequential(
            nn.Conv1d(dim, dim // 4, kernel_size=1, padding=0, stride=1),
            nn.BatchNorm1d(dim // 4),
            nn.ReLU(inplace=True),
        )
        self.filter = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
        )
        self.kv_embed = nn.Conv1d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio) if sr_ratio > 1 else nn.Identity()
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2)
        )
        self.proj = nn.Linear(dim + dim // 4, dim)


    def forward(self, x, L):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.view(B, L, C).permute(0, 2,1)
        x_dwt = self.dwt(self.reduce(x))
        x_dwt = self.filter(x_dwt)
        x_idwt = self.idwt(x_dwt)  # [32,16,56,56]
        x_idwt = x_idwt.view(B, -1, x_idwt.size(-1)).transpose(1, 2)
        kv = self.kv_embed(x_dwt).reshape(B, C, -1).permute(0, 2, 1)
        kv = self.kv(kv).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(torch.cat([x, x_idwt], dim=-1))
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio=1, drop_path=0.,block_type='wave'):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        if block_type == 'std_att':
            self.attn = Attention(dim, num_heads=num_heads)
        else:
            self.attn = WaveAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, L):
        # [32,3136,96] h=56,w=56
        x = x + self.drop_path(self.attn(self.norm1(x), L))  # pre_norm
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pa_conv = nn.Conv1d(dim, dim, kernel_size=11, padding=11//2, groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.sigmoid(self.pa_conv(x))

class ConvStem(nn.Module):
    def __init__(self, in_ch=1, out_ch=96, patch_size=4, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        stem = []
        in_dim, out_dim = in_ch, out_ch // 2
        for i in range(2):
            stem.append(nn.Conv1d(in_dim, out_dim, kernel_size=11, stride=2, padding=11//2, bias=False))
            stem.append(nn.BatchNorm1d(out_dim))
            stem.append(nn.ReLU(inplace=True))
            in_dim, out_dim = out_dim, out_dim * 2

        stem.append(nn.Conv1d(in_dim, out_ch, kernel_size=1, stride=1))
        self.proj = nn.Sequential(*stem)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)  # [32,32,4000]
        if self.with_pos:
            x = self.pos(x)  # [32,32,4000]
        x = x.transpose(1, 2)  # BCHW -> BNC [32,4000,32]
        x = self.norm(x)
        L = L // self.patch_size
        return x, L

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, out_ch=96, patch_size=2, with_pos=True):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=11, stride=patch_size, padding=11 // 2)

        self.with_pos = with_pos
        if self.with_pos:
            self.pos = PA(out_ch)

        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x):
        B, C, L = x.shape
        x = self.proj(x)
        if self.with_pos:
            x = self.pos(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        L = L // self.patch_size
        return x, L

class ResTV2(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, embed_dims=[96, 192, 384, 768],
                 num_heads=[1, 2, 4, 8], drop_path_rate=0.,
                 depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.stem = ConvStem(in_chans, embed_dims[0], patch_size=4)
        self.patch_2 = PatchEmbed(embed_dims[0], embed_dims[1], patch_size=2)
        self.patch_3 = PatchEmbed(embed_dims[1], embed_dims[2], patch_size=2)
        self.patch_4 = PatchEmbed(embed_dims[2], embed_dims[3], patch_size=2)

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], num_heads[0], sr_ratios[0], dpr[cur + i],block_type="wave")
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], num_heads[1], sr_ratios[1], dpr[cur + i],block_type="wave")
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], num_heads[2], sr_ratios[2], dpr[cur + i],block_type="std_att")
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], num_heads[3], sr_ratios[3], dpr[cur + i],block_type="std_att")
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)  # final norm layer
        # classification head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # input x->[32,1,16000]
        B, _, L = x.shape
        # x->[32,96,4000]->[32,4000,96] L=4000
        x, L = self.stem(x)
        # stage 1
        for blk in self.stage1:
            x = blk(x, L)  # output x->[32,3136,96]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 2
        x, L = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x, L)  # # output x->[32,784,192]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 3
        x, L = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x, L)  # [32,196,384]
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        # stage 4
        x, L = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x, L)  # [32,49,768]
        x = self.norm(x)
        x = x.permute(0, 2, 1).reshape(B, -1, L)
        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x

def restv2_DWT(pretrained=False, **kwargs):  # 82.3|4.7G|24M -> |3.92G|30.37M   4.5G|30.33M
    model = ResTV2(embed_dims=[32, 64, 128, 256], depths=[1, 2, 2, 2], **kwargs)
    return model
from thop.profile import profile
if __name__ == "__main__":
    input = torch.randn(32, 1, 16000)
    model = restv2_DWT(pretrained=False, num_classes=2)
    output = model(input)
    print(output.shape)  # [32,2]

    total_ops, total_params = profile(model, (input,), verbose=False)
    print(
        "%s | %.2f | %.2f" % ("resnet50", total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )