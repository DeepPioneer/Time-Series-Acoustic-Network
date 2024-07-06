import torch
import torch.nn as nn
from timm.models.layers import DropPath,trunc_normal_
import os,math
import numpy as np
import torch.nn.functional as F
import torchinfo
from thop.profile import profile
from audio_project.config import get_args_parser

parser = get_args_parser()
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:

        all_top_indices_x = [0, 1, 2, 4, 8, 10, 12, 14, 18, 24, 28, 32, 40, 48, 50,60]

        all_top_indices_x = [int(x / 32) for x in all_top_indices_x]  # 转换为整数
        mapper_x = all_top_indices_x[:num_freq]

    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        mapper_x = all_low_indices_x[:num_freq]

    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        mapper_x = all_bot_indices_x[:num_freq]

    else:
        raise NotImplementedError

    return mapper_x

class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, length, mapper_x, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init for 1D
        self.register_buffer('weight', self.get_dct_filter(length, mapper_x, channel))

    def forward(self, x):
        assert len(x.shape) == 3, 'x must be 3 dimensions (batch, channel, length), but got ' + str(len(x.shape))

        x = x * self.weight

        result = torch.sum(x, dim=2)
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, length, mapper_x, channel):
        # dct_filter->[64,4000]
        dct_filter = torch.zeros(channel, length)
        # c_part = 64 // 16 = 4
        c_part = channel // len(mapper_x)
        # mapper_x->[100, 200, 300, 400, 500, 600, 700, 800, 900, 1000,
        #                    1100, 1200, 1300, 1400, 1500, 1600]
        for i, u_x in enumerate(mapper_x):
            # print(f"u_x type: {type(u_x)}")  # 检查u_x的类型
            for t in range(length):
                dct_filter[i * c_part: (i + 1) * c_part, t] = self.build_filter(t, u_x, length)

        return dct_filter

class MultiSpectralAttentionLayer1D(nn.Module):
    def __init__(self, channel, length, reduction=16, freq_sel_method='top16'):
        super(MultiSpectralAttentionLayer1D, self).__init__()
        self.reduction = reduction
        self.length = length
        mapper_x = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        # 使用向下取整
        mapper_x = [temp_x * math.floor(length / 62) for temp_x in mapper_x]
        # make the frequencies in different sizes are identical to a 500 frequency space
        # eg, (2) in 1000 is identical to (1) in 500
        # 对于一维音频信号，我们仅需要一个mapper_x，因为只有时间维度
        self.dct_layer = MultiSpectralDCTLayer(length, mapper_x, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, l = x.shape
        if l != self.length:
            x = torch.nn.functional.adaptive_avg_pool1d(x, self.length)
        y = self.dct_layer(x)
        y = self.fc(y).view(n, c, 1)
        return x * y.expand_as(x)

def broadcast_dim(x):
    """
    Auto broadcast input so that it can fit into a Conv1d
    """
    if x.dim() == 2:
        x = x[:, None, :]
    elif x.dim() == 1:
        # If nn.DataParallel is used, this broadcast doesn't work
        x = x[None, :]
    elif x.dim() == 3:
        pass
    else:
        raise ValueError(
            "Only support input with shape = (batch, len) or shape = (len)"
        )
    return x

def create_dft_kernels(n_fft, freq_bins=None,fmin=50,fmax=6000,sr=44100):
    if freq_bins is None:
        freq_bins = (n_fft-1)//2 + 1
    s = np.arange(n_fft)
    wsin = np.zeros((freq_bins, n_fft))
    wcos = np.zeros((freq_bins, n_fft))

    bins2freq = []
    binslist = []

    start_bin = fmin * n_fft / sr
    scaling_ind = (fmax - fmin) * (n_fft / sr) / freq_bins

    for k in range(freq_bins):  # Only half of the bins contain useful info
        # print("linear freq = {}".format((k*scaling_ind+start_bin)*sr/n_fft))
        freq = (k * scaling_ind + start_bin) * sr / n_fft
        bins2freq.append(freq)
        binslist.append((k * scaling_ind + start_bin))
        wsin[k, :] = np.sin(2 * np.pi * freq * s / sr)
        wcos[k, :] = np.cos(2 * np.pi * freq * s / sr)

    return (wsin.astype(np.float32),wcos.astype(np.float32),bins2freq,binslist)

class DFT_func(nn.Module):
    def __init__(
            self,
            n_fft=2048,
            freq_bins=None,
            fmin=50,
            fmax=6000,
            sr=22050,
            trainable=False,
            output_format="Complex",
    ):
        super(DFT_func, self).__init__()

        self.output_format = output_format
        self.trainable = trainable
        self.n_fft = n_fft
        self.freq_bins = freq_bins
        # Create filter windows for dft
        (
            kernel_sin,
            kernel_cos,
            self.bins2freq,
            self.bin_list,
        ) = create_dft_kernels(
            n_fft,
            freq_bins=freq_bins,
            fmin=fmin,
            fmax=fmax,
            sr=sr,
        )

        kernel_sin = torch.tensor(kernel_sin, dtype=torch.float)
        kernel_cos = torch.tensor(kernel_cos, dtype=torch.float)

        if not self.trainable:
            self.register_buffer("wsin", kernel_sin)
            self.register_buffer("wcos", kernel_cos)
        else:
            self.wsin = nn.Parameter(kernel_sin, requires_grad=self.trainable)
            self.wcos = nn.Parameter(kernel_cos, requires_grad=self.trainable)

    def forward(self, x, output_format=None):
        output_format = output_format or self.output_format
        x = broadcast_dim(x)
        spec_imag = torch.matmul(x, self.wsin.permute(1, 0))
        spec_real = torch.matmul(x, self.wcos.permute(1, 0)) # Doing DFT by using matrix multiplication
        if output_format == "Magnitude":
            spec = spec_real.pow(2) + spec_imag.pow(2)
            if self.trainable:
                return torch.sqrt(spec + 1e-8)  # prevent Nan gradient when sqrt(0) due to output=0
            else:
                return torch.sqrt(spec)

        elif output_format == "Complex":
            complex_spec = torch.view_as_complex(torch.stack((spec_real, -spec_imag), -1))
            return complex_spec.squeeze()

        elif output_format == "Phase":
            return torch.atan2(-spec_imag + 0.0,
                               spec_real)  # +0.0 removes -0.0 elements, which leads to error in calculating phase

class ICB(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0.):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, hidden_features, 1)
        self.conv2 = nn.Conv1d(in_features, hidden_features, 3, 1, 1)
        self.conv3 = nn.Conv1d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)
        self.act = nn.GELU()
        c2wh = dict([(32, 4000), (64, 1000), (128, 250), (256, 62)])
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.MultiSpectralAttention = MultiSpectralAttentionLayer1D(hidden_features, c2wh[in_features],
                                                                    reduction=16,
                                                                    freq_sel_method='top16')

    def forward(self, x):
        x = x.transpose(1, 2)
        # x1 = self.conv1(x)
        x1 = self.MultiSpectralAttention(self.bn2(self.conv1(x)))
        x1_1 = self.act(x1)
        x1_2 = self.drop(x1_1)

        # x2 = self.conv2(x)
        x2 = self.MultiSpectralAttention(self.bn2(self.conv2(x)))
        x2_1 = self.act(x2)
        x2_2 = self.drop(x2_1)

        out1 = x1 * x2_2
        out2 = x2 * x1_2

        x = self.conv3(out1 + out2)
        x = x.transpose(1, 2)
        return x

class Adaptive_Spectral_Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.complex_weight_high = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)
        self.complex_weight = nn.Parameter(torch.randn(dim, 2, dtype=torch.float32) * 0.02)

        trunc_normal_(self.complex_weight_high, std=.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.threshold_param = nn.Parameter(torch.rand(1)) # * 0.5)

    def create_adaptive_high_freq_mask(self, x_fft):
        B, _, _ = x_fft.shape # torch.Size([32, 2000, 128])
        # Calculate energy in the frequency domain
        energy = torch.abs(x_fft).pow(2).sum(dim=-1)
        # Flatten energy across H and W dimensions and then compute median
        flat_energy = energy.view(B, -1)  # Flattening H and W into a single dimension
        median_energy = flat_energy.median(dim=1, keepdim=True)[0]  # Compute median
        median_energy = median_energy.view(B, 1)  # Reshape to match the original dimensions

        # Normalize energy
        epsilon = 1e-6  # Small constant to avoid division by zero
        normalized_energy = energy / (median_energy + epsilon)
        adaptive_mask = ((normalized_energy > self.threshold_param).float() - self.threshold_param).detach() + self.threshold_param
        adaptive_mask = adaptive_mask.unsqueeze(-1)

        return adaptive_mask

    def forward(self, x_in):
        B, N, C = x_in.shape
        # print("x_in",x_in.shape) # torch.Size([32, 3999, 128])
        dtype = x_in.dtype
        x = x_in.to(torch.float32)

        # Apply FFT along the time dimension
        # x_fft = torch.fft.rfft(x, dim=1, norm='ortho')
        x = x.permute(0, 2, 1)
        # Apply temporal background equalization
        window_size = 128  # You can choose the window size based on your requirements
        equalizer = TemporalBackgroundEqualizer(window_size=window_size)
        x = equalizer(x)
        method = DFT_func(n_fft=x.shape[2], freq_bins=None, fmin=50, fmax=6000, sr=16000,
                          trainable=True, output_format="Complex")
        x_fft = method(x, output_format="Complex")
        x_fft = x_fft.permute(0,2,1)
        # print(x_fft.shape) # torch.Size([32, 2000, 128])
        weight = torch.view_as_complex(self.complex_weight) # weight torch.Size([128])
        x_weighted = x_fft * weight

        if args.adaptive_filter:
            # Adaptive High Frequency Mask (no need for dimensional adjustments)
            freq_mask = self.create_adaptive_high_freq_mask(x_fft)
            x_masked = x_fft * freq_mask.to(x.device)

            weight_high = torch.view_as_complex(self.complex_weight_high)
            x_weighted2 = x_masked * weight_high
            x_weighted += x_weighted2
        # Apply Inverse FFT
        x = torch.fft.irfft(x_weighted, n=N, dim=1, norm='ortho')

        x = x.to(dtype)
        x = x.view(B, N, C)  # Reshape back to original shape

        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=3., drop=0., drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.asb = Adaptive_Spectral_Block(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.icb = ICB(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        # Check if both ASB and ICB are true
        if args.ICB and args.ASB:
            x = x + self.drop_path(self.icb(self.norm2(self.asb(self.norm1(x)))))
        # If only ICB is true
        elif args.ICB:
            x = x + self.drop_path(self.icb(self.norm2(x)))
        # If only ASB is true
        elif args.ASB:
            x = x + self.drop_path(self.asb(self.norm1(x)))
        # If neither is true, just pass x through
        return x

class PatchEmbed(nn.Module):
    def __init__(self,patch_size=8, in_chans=3, embed_dim=384):
        super().__init__()
        stride = patch_size // 2
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,padding=stride // 2)

    def forward(self, x):
        x_out = self.proj(x).flatten(2).transpose(1, 2)
        return x_out

class TemporalBackgroundEqualizer(nn.Module):
    def __init__(self, window_size):
        super().__init__()
        self.window_size = window_size
        # Initialize coef_small within [0, 1) and coef_large greater than 1
        self.coef_small = nn.Parameter(torch.rand(1))
        self.coef_large = nn.Parameter(torch.rand(1) + 1.0)

    def forward(self, x):
        # x shape: [batch, channels, length]
        batch, channels, length = x.size()

        # Compute moving average with a convolution
        kernel = torch.ones(channels, 1, self.window_size, device=x.device) / self.window_size
        moving_avg = F.conv1d(x, kernel, padding=self.window_size // 2, groups=channels)

        # Adjust moving_avg length to match input length
        if moving_avg.size(-1) > length:
            moving_avg = moving_avg[..., :length]
        elif moving_avg.size(-1) < length:
            moving_avg = F.pad(moving_avg, (0, length - moving_avg.size(-1)))

        # Compare and adjust the signal
        adjusted = torch.where(x > moving_avg, x * self.coef_small, x * self.coef_large)

        return adjusted

class ResTV2(nn.Module):
    def __init__(self, num_classes=5, embed_dims=[96, 192, 384, 768],drop_path_rate=0.,
                 depths=[2, 2, 2, 2]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        self.patch_1 = PatchEmbed(patch_size=8, in_chans=1, embed_dim=embed_dims[0])
        self.patch_2 = PatchEmbed(patch_size=8, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_3 = PatchEmbed(patch_size=8, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_4 = PatchEmbed(patch_size=8, in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        self.stage1 = nn.ModuleList([
            Block(embed_dims[0], drop=args.dropout_rate, drop_path=dpr[cur + i])
            for i in range(depths[0])
        ])

        cur += depths[0]
        self.stage2 = nn.ModuleList([
            Block(embed_dims[1], drop=args.dropout_rate, drop_path=dpr[cur + i])
            for i in range(depths[1])
        ])

        cur += depths[1]
        self.stage3 = nn.ModuleList([
            Block(embed_dims[2], drop=args.dropout_rate, drop_path=dpr[cur + i])
            for i in range(depths[2])
        ])

        cur += depths[2]
        self.stage4 = nn.ModuleList([
            Block(embed_dims[3], drop=args.dropout_rate, drop_path=dpr[cur + i])
            for i in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1], eps=1e-6)  # final norm layer
        # classification head
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

    def forward(self, x):  # input x->[32,1,16000]
        B, _, L = x.shape
        x = self.patch_1(x)
        # stage 1
        for blk in self.stage1:
            x = blk(x)  # output x->[32,3999,32]
        x = x.permute(0, 2, 1)
        # stage 2
        x = self.patch_2(x)
        for blk in self.stage2:
            x = blk(x)  #  torch.Size([32, 998, 64])
        x = x.permute(0, 2, 1)
        # stage 3
        x = self.patch_3(x)
        for blk in self.stage3:
            x = blk(x)  # torch.Size([32, 248, 128])
        x = x.permute(0, 2, 1)
        # stage 4
        x = self.patch_4(x)
        for blk in self.stage4:
            x = blk(x)  # [32,61,256]
        x = self.norm(x)
        x = x.permute(0, 2, 1)
        x = self.avg_pool(x).flatten(1)
        x = self.head(x)
        return x

def TS_ASICNet(pretrained=False,**kwargs):  # 82.3|4.7G|24M -> |3.92G|30.37M   4.5G|30.33M
    model = ResTV2(embed_dims=[32, 64, 128, 256], depths=[1, 2, 2, 2], **kwargs)
    return model

if __name__ == "__main__":
    input = torch.randn(32, 1, 16000)
    model = TS_ASICNet()
    output = model(input)
    print(output.shape)  # [32,2]
    total_ops, total_params = profile(model, (input,), verbose=False)
    print(
        "%s | %.2f | %.2f" % ("Fca_WaveMsNet", total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )
