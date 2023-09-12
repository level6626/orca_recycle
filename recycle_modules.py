"""
This module contains the recycling modules which 
are components of the Orca models.
"""
import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from orca_modules import Net


class MultiplicativeEncoder(nn.Module):
    def __init__(self, dim, n_mats, length):
        """
        Parameters
        ----------
        Mat : n_mats x Length x Length
        """
        super(MultiplicativeEncoder, self).__init__()
        self.dim = dim
        self.n_mats = n_mats
        self.length = length

        self.initial_x = nn.Parameter(torch.randn(1, dim, length))

        self.linear = nn.ModuleList(
            [nn.Conv1d(dim, dim * n_mats, kernel_size=1) for _ in range(6)]
        )

        self.lms = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.BatchNorm1d(dim))
                for _ in range(6)
            ]
        )
        self.ms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.BatchNorm1d(dim),
                    nn.ReLU(),
                )
                for _ in range(6)
            ]
        )

    def forward(self, mat):
        x = self.initial_x
        for l, lm, m in zip(self.linear, self.lms, self.ms):
            x = lm(
                x
                + torch.matmul(
                    l(x).view(-1, self.n_mats, self.dim, self.length), mat
                ).sum(1)
            )
            x = x + m(x)
        return x


class MultiplicativeEncoderGnorm(MultiplicativeEncoder):
    def __init__(self, dim, n_mats, length):
        """
        Parameters
        ----------
        Mat : n_mats x Length x Length
        """
        super(MultiplicativeEncoderGnorm, self).__init__(
            dim=dim, n_mats=n_mats, length=length
        )
        self.lms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=1), nn.GroupNorm(1, dim)
                )
                for _ in range(6)
            ]
        )
        self.ms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GroupNorm(1, dim),
                    nn.ReLU(),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GroupNorm(1, dim),
                    nn.ReLU(),
                )
                for _ in range(6)
            ]
        )


class MultiplicativeFF(nn.Module):
    def __init__(self, dim, n_mats, lognormmat):
        """
        Parameters
        ----------

        """
        super(MultiplicativeFF, self).__init__()
        self.dim = dim
        self.n_mats = n_mats
        # self.length = length
        self.register_buffer("lognormmat", lognormmat)
        self.weight = nn.Parameter(torch.ones(n_mats))
        self.softplus = nn.Softplus()

        self.linear = nn.ModuleList(
            [nn.Conv1d(dim, dim * self.n_mats, kernel_size=1) for _ in range(6)]
        )

        self.lms = nn.ModuleList(
            [
                nn.Sequential(nn.Conv1d(dim, dim, kernel_size=1), nn.GroupNorm(16, dim))
                for _ in range(12)
            ]
        )
        self.ms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GroupNorm(16, dim),
                    # nn.Dropout(),
                    nn.ReLU(),
                    nn.Conv1d(dim, dim, kernel_size=1),
                    nn.GroupNorm(16, dim),
                    # nn.Dropout(),
                    nn.ReLU(),
                )
                for _ in range(12)
            ]
        )

    def forward(self, x, mat):
        """
        mat : n_mats x Length x Length
        """
        length = mat.shape[-1]
        mat = torch.exp(
            self.softplus(self.weight)[None, :, None, None]
            * (mat + self.lognormmat[None, :, :, :].expand(mat.shape))
        )
        for l, lm, m in zip(self.linear, self.lms, self.ms):
            x = lm(
                x
                + torch.matmul(l(x).view(-1, self.n_mats, self.dim, length), mat).sum(1)
            )
            x = x + m(x)
        return x


class NetGnorm(nn.Module):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model. The trained model weighted can be
        loaded into Encoder and Decoder_1m modules.

        Parameters
        ----------
        num_1d : int or None, optional
            The number of 1D targets used for the auxiliary
            task of predicting ChIP-seq profiles.
        """
        super(NetGnorm, self).__init__()

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
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.GroupNorm(1, 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )
        if num_1d is not None:
            self.final_1d = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.GroupNorm(1, 128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, num_1d, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        self.num_1d = num_1d

    def forward(self, x):
        """Forward propagation of a batch."""

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
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, dummy)
        else:
            cur = checkpoint(run0, x, dummy)

        def run1(cur):
            first = True
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


class NetRcyc1d(Net):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model with recycling mechanism
        """
        super(NetRcyc1d, self).__init__(num_1d=num_1d)
        self.mEncoder = MultiplicativeEncoder(dim=128, n_mats=1, length=250)

    def forward(self, x, mat=None):
        """Forward propagation of a batch."""

        def run0(x, mat, dummy):
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
            if mat is not None:
                mat1d = self.mEncoder(mat)
                out7 = self.conv7(lout7 + mat1d)
            else:
                out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, mat, dummy)
        else:
            cur = checkpoint(run0, x, mat, dummy)

        def run1(cur):
            first = True
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


class NetRcyc2d(Net):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model with recycling mechanism
        """
        super(NetRcyc2d, self).__init__(num_1d=num_1d)
        self.mconv = nn.Conv2d(1, 128, kernel_size=(1, 1), padding=0)

    def forward(self, x, mat=None):
        """Forward propagation of a batch."""

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
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, dummy)
        else:
            cur = checkpoint(run0, x, dummy)

        def run1(cur, mat):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, mat)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


class NetRcyc1d2d(Net):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model with recycling mechanism
        """
        super(NetRcyc1d2d, self).__init__(num_1d=num_1d)
        self.mconv = nn.Conv2d(1, 128, kernel_size=(1, 1), padding=0)
        self.mEncoder = MultiplicativeEncoder(dim=128, n_mats=1, length=250)
        self.upSample = nn.Upsample(scale_factor=250)

    def forward(self, x, mat=None):
        """Forward propagation of a batch."""

        def run0(x, mat, dummy):
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
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, mat, dummy)
        else:
            cur = checkpoint(run0, x, mat, dummy)

        def run1(cur, mat):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, mat)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


class NetGnormRcyc1d2d(NetGnorm):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model with recycling mechanism
        """
        super(NetGnormRcyc1d2d, self).__init__(num_1d=num_1d)
        self.mconv = nn.Conv2d(1, 128, kernel_size=(1, 1), padding=0)
        self.mEncoder = MultiplicativeEncoderGnorm(dim=128, n_mats=1, length=250)
        self.upSample = nn.Upsample(scale_factor=250)

    def forward(self, x, mat=None):
        """Forward propagation of a batch."""

        def run0(x, mat, dummy):
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
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, mat, dummy)
        else:
            cur = checkpoint(run0, x, mat, dummy)

        def run1(cur, mat):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur, mat)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur


class NetRcyc1d2df(Net):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model with recycling mechanism
        """
        super(NetRcyc1d2df, self).__init__(num_1d=num_1d)
        self.mconv = nn.Conv2d(1, 128, kernel_size=(1, 1), padding=0)
        self.mEncoder = MultiplicativeEncoder(dim=128, n_mats=1, length=250)
        self.upSample = nn.Upsample(scale_factor=250)

    def forward(self, x, mat=None):
        """Forward propagation of a batch."""

        def run0(x, mat, dummy):
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
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = run0(x, mat, dummy)
        else:
            cur = run0(x, mat, dummy)

        def run1(cur, mat, dummy):
            first = True
            if mat is not None:
                cur = cur + self.mconv(mat)
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur, dummy):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur, dummy):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = run1(cur, mat, dummy)
        cur = run2(cur, dummy)
        cur = run3(cur, dummy)

        if self.num_1d:
            return cur, output1d
        else:
            return cur

class Netf(Net):
    def __init__(self, num_1d=None):
        """
        frozen Orca 1Mb model
        """
        super(Netf, self).__init__(num_1d=num_1d)

    def forward(self, x):
        """Forward propagation of a batch."""

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
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = run0(x, dummy)
        else:
            cur = run0(x, dummy)

        def run1(cur, dummy):
            first = True
            for lm, m in zip(self.lconvtwos[:7], self.convtwos[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur, dummy):
            for lm, m in zip(self.lconvtwos[7:13], self.convtwos[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur, dummy):
            for lm, m in zip(self.lconvtwos[13:], self.convtwos[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = run1(cur, dummy)
        cur = run2(cur, dummy)
        cur = run3(cur, dummy)

        if self.num_1d:
            return cur, output1d
        else:
            return cur