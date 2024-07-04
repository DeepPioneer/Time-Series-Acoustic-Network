import torch
import torch.nn as nn
import numpy as np
import math
import torchinfo
from thop.profile import profile

device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:

        all_top_indices_x = [0, 2, 4, 8, 16, 24, 32, 48, 64, 80, 96, 102,
                             128, 140,160,200]

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
        mapper_x = [temp_x * math.floor(length / 500) for temp_x in mapper_x]
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


# 音频软阈值收缩法
class audio_Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(audio_Shrinkage, self).__init__()
        #         print(channel)
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  # CS
        # average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class acoustic_Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(acoustic_Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)  # torch.Size([16, 72, 1])
        x = torch.flatten(x, 1)  # # torch.Size([16, 72])
        alpha = self.fc(x)  # torch.size([16,72])
        # 动态调整范围
        # 计算每一行的中位数绝对偏差
        median_values = torch.median(x_abs, dim=-1, keepdim=True).values
        mad_values = torch.median(torch.abs(x_abs - median_values), dim=-1).values
        # 将每一行的中位偏差乘以系数
        T = mad_values * alpha
        T = T.unsqueeze(2)
        # soft thresholding
        sub = x_abs - T
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


# 水声自适应软阈值背景均衡收缩法
class acoustic_Shrinkage_new(nn.Module):
    def __init__(self, channel, gap_size):
        super(acoustic_Shrinkage_new, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = 20 * torch.log10(x + 1)
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)  # torch.Size([16, 72, 1])
        x = torch.flatten(x, 1)  # # torch.Size([16, 72])
        alpha = self.fc(x)  # torch.size([16,72])
        # 动态调整范围
        sorted_signal, indices = torch.sort(x_abs)
        media_signal = torch.median(sorted_signal, dim=2).values
        # Compute maxima and minima of each row
        row_maxima = torch.max(sorted_signal, dim=2).values
        row_minima = torch.min(sorted_signal, dim=2).values
        # Calculate differences
        diff_max_median = row_maxima - media_signal
        diff_median_min = media_signal - row_minima
        # Adjust median values
        T = torch.where(diff_max_median > diff_median_min, media_signal + alpha * (row_minima - media_signal),
                        media_signal + alpha * (row_maxima - media_signal))
        T = T.unsqueeze(2)
        # soft thresholding
        sub = x_abs - T
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(DepthwiseSeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, use_depthwise=False,
                 *, reduction=16):
        super(BasicBlock, self).__init__()
        c2wh = dict([(64, 4000), (128, 2000), (256, 1000), (512, 500)])
        if use_depthwise:
            self.conv1 = DepthwiseSeparableConv1d(in_channels, out_channels, kernel_size=3, stride=stride,
                                                  padding=1)
            self.conv2 = DepthwiseSeparableConv1d(out_channels, out_channels, kernel_size=3, padding=1)
            # 软阈值降噪
            self.shrinkage = acoustic_Shrinkage_new(out_channels, gap_size=(1))  # acoustic_Shrinkage audio_Shrinkage
        else:
            self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
            # 软阈值降噪
            self.shrinkage = acoustic_Shrinkage_new(out_channels, gap_size=(1))  # acoustic_Shrinkage audio_Shrinkage

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.MultiSpectralAttention = MultiSpectralAttentionLayer1D(out_channels, c2wh[out_channels],
                                                                    reduction=reduction,
                                                                    freq_sel_method='top16')
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                DepthwiseSeparableConv1d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.MultiSpectralAttention(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class RSNet(nn.Module):

    def __init__(self, block, num_block, num_classes=5):
        super().__init__()

        self.in_channels = 1

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(self.in_channels),
            nn.ReLU(inplace=True)
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, use_depthwise=True)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2, use_depthwise=True)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, use_depthwise=True)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, use_depthwise=True)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1, use_depthwise=False):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride, use_depthwise=use_depthwise))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = x.unsqueeze(1)
        output = self.conv1(x)
        output = self.maxpool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output

def rsnet18(num_classes):
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2])
    # return RSNet(BasicBlock, [3, 4, 6, 3],num_classes)

class WaveMsNet(nn.Module):
    def __init__(self, num_classes=5):
        super(WaveMsNet, self).__init__()

        self.RecModel = rsnet18(num_classes)

    def forward(self, x):
        h = self.RecModel(x)
        return h


if __name__ == "__main__":
    input_data = torch.randn(32, 1, 16000)
    input_data = input_data.to(device)

    model = Fca_WaveMsNet(num_classes=5).to(device)

    output = model(input_data)

    print(output.shape)
    # 打印网络参数数量
    print("Num Parameters:", sum([p.numel() for p in model.parameters()]))
    # summary(model,input_size=(1,1690),batch_size=batch_size)
    # print(torchinfo.summary(model, (32,1,16000), batch_dim=0,
    #                         col_names=('input_size', 'output_size', 'num_params', 'kernel_size', 'mult_adds'),
    #                         verbose=0))

    total_ops, total_params = profile(model, (input_data,), verbose=False)
    print(
        "%s | %.2f | %.2f" % ("Fca_WaveMsNet", total_params / (1000 ** 2), total_ops / (1000 ** 3))
    )

