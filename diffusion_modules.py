import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from recycle_modules import MultiplicativeEncoderGnorm

class GaussianFourierProjection(nn.Module):
    """
    Gaussian random features for encoding time steps.
    """

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense1D(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


class Dense2D(nn.Module):
    """
    A fully connected layer that reshapes outputs to feature maps.
    """

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
    def __init__(self, marginal_prob_std, channels=[32, 64, 128], embed_dim=128, num_1d=None):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super(ScoreNet, self).__init__()
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
        )
        self.dense1 = Dense1D(embed_dim, 64)

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.ReLU(inplace=True),
        )
        self.dense2 = Dense1D(embed_dim, 96)

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
        self.dense3 = Dense1D(embed_dim, 128)

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
        self.dense4 = Dense1D(embed_dim, 128)

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
        self.dense5 = Dense1D(embed_dim, 128)

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
        self.dense6 = Dense1D(embed_dim, 128)

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )

        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                ),
            ]
        )

        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.GroupNorm(1, 64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.denses = nn.ModuleList(
            [Dense2D(embed_dim, 64) for _ in range(19)]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.GroupNorm(1, 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )
        self.act = nn.ReLU()
        self.marginal_prob_std = marginal_prob_std
        self.mconv = nn.Conv2d(1, 128, kernel_size=(1, 1), padding=0)
        self.mEncoder = MultiplicativeEncoderGnorm(dim=128, n_mats=1, length=250)
        self.upSample = nn.Upsample(scale_factor=250)

    def forward(self, x, mat, t):
        """Forward propagation of a batch."""

        def run0(x, mat, t, dummy):
            embed = self.act(self.embed(t))
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            out1 = out1 + self.dense1(embed)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            out2 = out2 + self.dense2(embed)
            lout3 = self.lconv3(out2 + lout2)
            if mat is not None:
                mat1d = self.mEncoder(mat)
                mat1d = self.upSample(mat1d)
                out3 = self.conv3(lout3 + mat1d)
            else:
                out3 = self.conv3(lout3)
            out3 = out3 + self.dense3(embed)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            out4 = out4 + self.dense4(embed)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            out5 = out5 + self.dense5(embed)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            out6 = out6 + self.dense6(embed)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            return cur, embed

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        cur, embed = checkpoint(run0, x, mat, t, dummy)

        def run1(cur, mat, embed):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m, d in zip(self.lconvtwos[:7], self.convtwos[:7], self.denses[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run2(cur, embed):
            for lm, m, d in zip(self.lconvtwos[7:13], self.convtwos[7:13], self.denses[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run3(cur, embed):
            for lm, m, d in zip(self.lconvtwos[13:], self.convtwos[13:], self.denses[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, mat, embed)
        cur = checkpoint(run2, cur, embed)
        cur = checkpoint(run3, cur, embed)
        
        cur = cur / self.marginal_prob_std(t)[:, None, None, None]

        return cur


class ScoreNet_a(ScoreNet):
    '''
    Directly predict the target
    '''
    def __init__(self, marginal_prob_std, channels=[32, 64, 128], embed_dim=128, num_1d=None):
        super().__init__(marginal_prob_std, channels, embed_dim, num_1d)
    
    def forward(self, x, mat, t):
        cur = super().forward(x, mat, t)
        cur = self.marginal_prob_std(t)[:, None, None, None] * cur
        return cur

class ScoreNet_1d(ScoreNet):
    '''
    Add 1d chromatin profile guidance.
    Remove time embeding before addition of 2d matrices
    '''
    def __init__(self, marginal_prob_std, channels=[32, 64, 128], embed_dim=128, num_1d=None):
        super().__init__(marginal_prob_std, channels, embed_dim, num_1d)
        if num_1d is not None:
            self.final_1d = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.GroupNorm(1, 128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, num_1d, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        self.num_1d = num_1d
    
    def forward(self, x, mat, t):
        """Forward propagation of a batch."""

        def run0(x, mat, t, dummy):
            embed = self.act(self.embed(t))
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            if mat is not None:
                mat1d = self.mEncoder(mat)
                mat1d = self.upSample(mat1d)
                out3 = self.conv3(lout3 + mat1d)
            else:
                out3 = self.conv3(lout3)
            out3 = out3 + self.dense3(embed)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            out4 = out4 + self.dense4(embed)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            out5 = out5 + self.dense5(embed)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            out6 = out6 + self.dense6(embed)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d, embed
            else:
                return cur, embed

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d, embed = checkpoint(run0, x, mat, t, dummy)
        else:
            cur, embed = checkpoint(run0, x, mat, t, dummy)

        def run1(cur, mat, embed):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m, d in zip(self.lconvtwos[:7], self.convtwos[:7], self.denses[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run2(cur, embed):
            for lm, m, d in zip(self.lconvtwos[7:13], self.convtwos[7:13], self.denses[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run3(cur, embed):
            for lm, m, d in zip(self.lconvtwos[13:], self.convtwos[13:], self.denses[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, mat, embed)
        cur = checkpoint(run2, cur, embed)
        cur = checkpoint(run3, cur, embed)
        
        if self.num_1d:
            return cur, output1d
        else:
            return cur


class SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.GroupNorm(1, 64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.GroupNorm(1, 96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.GroupNorm(1, 128),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        def run0(x, dummy):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            return out7

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        cur = checkpoint(run0, x, dummy)
        
        return cur


class TimeEmbedHicEncoder(MultiplicativeEncoderGnorm):
    '''
    Add time embedding
    '''
    def __init__(self, dim, n_mats, length, embed_dim=128):
        """
        Parameters
        ----------
        Mat : n_mats x Length x Length
        """
        super().__init__(dim, n_mats, length)
        self.embed_dim = embed_dim
        self.denses = nn.ModuleList(
            [Dense1D(embed_dim, dim) for _ in range(6)]
        )

    def forward(self, mat, embed):
        x = self.initial_x
        for l, lm, m, d in zip(self.linear, self.lms, self.ms, self.denses):
            x = lm(
                x
                + torch.matmul(
                    l(x).view(-1, self.n_mats, self.dim, self.length), mat
                ).sum(1)
            )
            x = x + m(x)
            x = x + d(embed)
        return x


class ScoreNet_b(ScoreNet):
    '''
    Predict score
    No time embedding in Seq-encoder
    Add time embeding in Hic-encoder
    '''
    def __init__(self, seqEncoder, marginal_prob_std, channels=[32, 64, 128], embed_dim=128, num_1d=None):
        super().__init__(marginal_prob_std, channels, embed_dim, num_1d)
        self.seqEncoder = seqEncoder
        self.mEncoder = TimeEmbedHicEncoder(dim=128, n_mats=1, length=250, embed_dim=128)
        self.mconv = nn.Conv1d(256, 128, kernel_size=1, padding=0)
    
    def forward(self, x, mat, t):
        cur = self.seqEncoder(x)
        
        embed = self.act(self.embed(t))
        tcur = self.mEncoder(mat, embed)
        cur = torch.cat([cur, tcur], 1)
        cur = self.mconv(cur)
        cur = cur[:, :, :, None] + cur[:, :, None, :]
        
        def run1(cur, embed):
            first = True
            for lm, m, d in zip(self.lconvtwos[:7], self.convtwos[:7], self.denses[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run2(cur, embed):
            for lm, m, d in zip(self.lconvtwos[7:13], self.convtwos[7:13], self.denses[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur
            return cur

        def run3(cur, embed):
            for lm, m, d in zip(self.lconvtwos[13:], self.convtwos[13:], self.denses[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
                cur = d(embed) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, embed)
        cur = checkpoint(run2, cur, embed)
        cur = checkpoint(run3, cur, embed)
        
        return cur
        