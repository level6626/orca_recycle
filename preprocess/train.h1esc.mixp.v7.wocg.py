import sys
import os

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

import time
import cooler
from cooltools.lib.numutils import adaptive_coarsegrain

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau

import selene_sdk
from selene_sdk.targets import Target
from selene_sdk.samplers.dataloader import SamplerDataLoader
from selene_sdk.samplers import RandomPositionsSampler

sys.path.append("..")
from selene_utils2 import *

seed = 314

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 


def generate_matrix(mat_raw, mat_bal, normmat250, eps, isCg, isLog=True):
    if isCg:
        mat_cg = adaptive_coarsegrain(mat_bal, mat_raw)
        mat_bal = mat_cg

    mat250 = np.nanmean(np.nanmean(np.reshape(mat_bal, (250, 4, 250, 4)), axis=3), axis=1)

    if isLog:
        mat_logb = np.log(mat250 + eps) - np.log(normmat250 + eps)
    else:
        mat_logb = np.sqrt(mat250 / normmat250)

    return mat_logb


class HicSubTarFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, sub_prob,
                 sub_input_path, tar_input_path, normmat, features, shape,
                 filt_tresh=0.7, cg_sub=True, cg_tar=True, islog_sub=True, islog_tar=True):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.sub_prob = sub_prob

        self.sub_input_path = sub_input_path
        self.tar_input_path = tar_input_path
        self.initialized = False

        self.normmat250 = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        self.eps = np.min(self.normmat250)

        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])

        self.shape = shape

        self.filt_tresh = filt_tresh

        self.cg_sub = cg_sub
        self.cg_tar = cg_tar
        self.islog_sub = islog_sub
        self.islog_tar = islog_tar

    def get_feature_data(self, chrom, start, end):
        if not self.initialized:
            self.sub_data = cooler.Cooler(self.sub_input_path)
            self.tar_data = cooler.Cooler(self.tar_input_path)
            self.initialized = True

        self.chrom = chrom
        self.start = start
        self.end = end

        mat_t_bal = self.tar_data.matrix(balance=True).fetch((chrom, start, end))

        if np.mean(np.isnan(mat_t_bal)) >= self.filt_tresh:
            return None

        mat_t_raw = self.tar_data.matrix(balance=False).fetch((chrom, start, end))

        mat_si_raw = self.sub_data.matrix(balance=False).fetch((chrom, start, end))
        mat_st_raw = mat_t_raw - mat_si_raw

        vec_w = self.sub_data.bins().fetch((chrom, start, end))['weight'].astype(np.float32)
        mat_w = np.outer(vec_w, vec_w)
        mat_si_bal = mat_si_raw * mat_w
        mat_st_bal = mat_st_raw * mat_w

        mat_si_logb = generate_matrix(mat_si_raw, mat_si_bal, self.normmat250, self.eps,
                                      isCg=self.cg_sub, isLog=self.islog_sub)
        mat_st_logb = generate_matrix(mat_st_raw, mat_st_bal, self.normmat250, self.eps,
                                      isCg=self.cg_sub, isLog=self.islog_sub)
        mat_t_logb = generate_matrix(mat_t_raw, mat_t_bal, self.normmat250, self.eps,
                                     isCg=self.cg_tar, isLog=self.islog_tar)

        # sub_inp_mat = (sub_data * sub_weights / sub_normmat)
        # sub_tar_mat = ((whole_data - sub_inp_data) * sub_weights / sub_normmat)
        # whole_mat = whole_data * whole_weights / normmat

        return [mat_si_logb, mat_st_logb, mat_t_logb]


class HicAutoSubFeatures(Target):
    """
    #Accept a list of cooler files as input.
    """

    def __init__(self, subsub_prob, input_path, sub_normmat, features, shape, filt_tresh=0.7,
                 cg_sub=True, cg_tar=True, islog_sub=True, islog_tar=True):
        """
        Constructs a new `GenomicFeatures` object.
        """
        self.subsub_prob = subsub_prob

        self.input_path = input_path
        self.initialized = False

        self.sb_normmat250 = np.reshape(sub_normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        self.sb_eps = np.min(self.sb_normmat250)

        self.ssb_normmat250 = self.sb_normmat250 * subsub_prob  # np.reshape(sub_normat * subsub_prob, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
        self.ssb_eps = np.min(self.ssb_normmat250)

        self.n_features = len(features)
        self.feature_index_dict = dict([(feat, index) for index, feat in enumerate(features)])

        self.shape = shape

        self.filt_tresh = filt_tresh
        self.cg_sub = cg_sub
        self.cg_tar = cg_tar
        self.islog_sub = islog_sub
        self.islog_tar = islog_tar

    def get_feature_data(self, chrom, start, end):
        if not self.initialized:
            self.data = cooler.Cooler(self.input_path)
            self.initialized = True

        self.chrom = chrom
        self.start = start
        self.end = end

        mat_st_bal = self.data.matrix(balance=True).fetch((chrom, start, end))

        if np.mean(np.isnan(mat_st_bal)) >= self.filt_tresh:
            return None

        mat_st_raw = self.data.matrix(balance=False).fetch((chrom, start, end))

        mat_ssi_raw = np.random.binomial(mat_st_raw, self.subsub_prob)
        mat_sst_raw = mat_st_raw - mat_ssi_raw

        vec_w = self.data.bins().fetch((chrom, start, end))['weight'].astype(np.float32)
        mat_w = np.outer(vec_w, vec_w)
        mat_ssi_bal = mat_ssi_raw * mat_w
        mat_sst_bal = mat_sst_raw * mat_w

        # sub_inp_mat = (sub_data * sub_weights / sub_normmat)
        # subsub_inp_mat =  subsub_data * sub_weights / (sub_normat * 0.5)
        # subsub_tar_mat = (sub_data - subsub_data) * sub_weights / (sub_normmat* 0.5)

        mat_ssi_logb = generate_matrix(mat_ssi_raw, mat_ssi_bal, self.ssb_normmat250, self.ssb_eps,
                                       isCg=self.cg_sub, isLog=self.islog_sub)
        mat_sst_logb = generate_matrix(mat_sst_raw, mat_sst_bal, self.ssb_normmat250, self.ssb_eps,
                                       isCg=self.cg_sub, isLog=self.islog_sub)
        mat_st_logb = generate_matrix(mat_st_raw, mat_st_bal, self.sb_normmat250, self.sb_eps,
                                      isCg=self.cg_tar, isLog=self.islog_tar)

        return [mat_ssi_logb, mat_sst_logb, mat_st_logb]


class HicEncoder(nn.Module):
    def __init__(self, dim, n_mats, length):
        """
        Parameters
        ----------
        Mat : 1 x Length x Length
        """
        super(HicEncoder, self).__init__()
        self.dim = dim
        self.n_mats = n_mats
        self.length = length

        self.initial_x = nn.Parameter(torch.randn(1, dim, length))

        self.linear = nn.ModuleList([
            nn.Conv1d(dim, dim * n_mats, kernel_size=1) for _ in range(6)
        ])

        self.lms = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=1),
                # nn.GroupNorm(1, dim)
                nn.BatchNorm1d(dim),
            ) for _ in range(6)

        ])

        self.ms = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Conv1d(dim, dim, kernel_size=1),
                nn.BatchNorm1d(dim),
                nn.ReLU()
            ) for _ in range(6)
        ])

    def forward(self, mat):
        x = self.initial_x

        for l, lm, m in zip(self.linear, self.lms, self.ms):
            x = lm(x + torch.matmul(l(x).view(-1, self.n_mats, self.dim, self.length), mat).sum(1))
            x = x + m(x)

        return x


class SeqEncoder(nn.Module):
    def __init__(self):
        super(SeqEncoder, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, seq):
        x = seq

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


class Decoder(nn.Module):
    def __init__(self):
        """
        Orca 1Mb model. The trained model weighted can be
        loaded into Encoder and Decoder_1m modules.

        Parameters
        ----------
        num_1d : int or None, optional
            The number of 1D targets used for the auxiliary
            task of predicting ChIP-seq profiles.
        """
        super(Decoder, self).__init__()

        self.lconvtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )

        self.convtwos = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )

    def forward(self, cur):
        """Forward propagation of a batch."""

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
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)  # Diff here
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        return cur

class Net(nn.Module):
    def __init__(self, seqEncoder, hicEncoder, decoder):
        super(Net, self).__init__()
        self.seq_encoder = seqEncoder
        self.hic_encoder = hicEncoder
        self.decoder = decoder

    def forward(self, seq, subdata, mask, ignore_seq, ignore_sub):

        if not ignore_seq:
            scur = self.seq_encoder(seq)

        if not ignore_sub:
            tcur = self.hic_encoder(torch.cat([subdata.unsqueeze(1), mask.unsqueeze(1)], 1))

        if ignore_seq:
            cur = tcur
        elif ignore_sub:
            cur = scur
        else:
            cur = scur + tcur

        cur = cur[:, :, :, None] + cur[:, :, None, :]  # Diff here

        cur = self.decoder(cur)

        return cur

def figshow(x, np=False):
    if np:
        plt.imshow(x.squeeze())
    else:
        plt.imshow(x.squeeze().cpu().detach().numpy())
    plt.show()

def sqrtToLogScale(mat, fnormmat, snormmat, seps):
    p = (mat ** 2) * fnormmat
    return np.log(p + seps) - np.log(snormmat + seps)

if __name__ == "__main__":
    learning_rate = 0.002
    momentum = 0.98

    LOSS_NAME = "MSE"
    SUBSUB_DATA_PROB = 0.5
    ignore_sequence = True
    ignore_subdata = False
    load_decoder = True
    out_decoder_name = ""

    batch_size = 16
    num_workers = 6

    val_size = 16
    val_batch_size = 16
    val_num_workers = 16

    window_size = 1000 * 1000 # 1000000

    loss_output = 100  # 500
    validation_output = 1000
    total_iterations = 100000

    os.makedirs("./models/", exist_ok=True)
    os.makedirs("./png/", exist_ok=True)

    hic_file = "../resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"
    
    SUB_DATA_PROB = 1.0
    subhic_file = "../resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"
    normat_file = "../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res1000.npy"

    ref_g_file_fa = "../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa"
    ref_g_file_mmap = "../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap"

    normmat_bydist = np.exp(np.load(normat_file))[:1000]
    normmat = normmat_bydist[np.abs(np.arange(1000)[:, None] - np.arange(1000)[None, :])]
    normmat250 = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
    eps250 = np.min(normmat250)

    modelstr = f"h1esc.r1000.p{SUB_DATA_PROB}.wocg"

    if ignore_sequence and ignore_subdata:
        print("Seq or subdata should be not ignored together")
        exit(1)

    if ignore_sequence:
        modelstr += ".noseq"

    if ignore_subdata:
        modelstr += ".nosub"

    out = open(f"log_{modelstr}.txt", 'w+')

    print(f"Description of model {modelstr}.\n Loss: {LOSS_NAME},\t algo: SDG")
    out.write(f"Description of model {modelstr}.\n Loss: {LOSS_NAME},\t algo: SDG\n")
    #
    # print(f"{subhic_file}")
    # out.write(f"{subhic_file}\n")

    print(f"Learning rate: {learning_rate}")
    out.write(f"Learning rate: {learning_rate}\n")

    print(f"Batch size: {batch_size}, window size: {window_size}")
    out.write(f"Batch size: {batch_size}, window size: {window_size}\n")

    print("Creating or loading neural net")
    out.write("Creating or loading neural net\n")

    if ignore_sequence:
        seqPart = None
    else:
        try:
            seqPart = SeqEncoder()
            seqPart.load_state_dict(
                torch.load("./models/seqpart." + modelstr + ".checkpoint")
            )
            print("Seq Model succefully loaded")
        except:
            print("NO SAVED Sequence part of model found!")
            seqPart = SeqEncoder()

    if ignore_subdata:
        hicPart = None
    else:
        try:
            hicPart = HicEncoder(128, 2, 250)
            hicPart.load_state_dict(
                torch.load("./models/hicpart." + modelstr + ".checkpoint")
            )
            print("Hic Model succefully loaded")
        except:
            print("NO SAVED HIC part of model found!")
            hicPart = HicEncoder(128, 2, 250)

    if load_decoder:
        try:
            decoder = Decoder()
            decoder.load_state_dict(
                torch.load("./models/decoderpart." + out_decoder_name + "." + modelstr + ".checkpoint")
            )
            print("Decoder Model succefully loaded")
        except:
            print("NO SAVED DECODER part of model found!")
            decoder = Decoder()
    else:
        decoder = Decoder()

    net = nn.DataParallel(
        Net(seqEncoder=seqPart, hicEncoder=hicPart, decoder=decoder)
    )

    net.cuda()
    net.train()

    # generate validation set
    print("Generate validation")
    out.write("Generate validation\n")

    val_sampler = RandomPositionsSampler(
        reference_sequence=MemmapGenome(
            input_path=ref_g_file_fa,
            memmapfile=ref_g_file_mmap,
            blacklist_regions='hg38'
        ),
        target=HicSubTarFeatures(SUB_DATA_PROB, subhic_file, hic_file, normmat,
                                 ['r1000'], [(250, 250), (250, 250), (250, 250)], filt_tresh=0.7,
                                 cg_sub=False, cg_tar=False, islog_sub=True, islog_tar=True),
        features=['r1000'],
        test_holdout=['chr8'],
        validation_holdout=['chr9'],
        sequence_length=window_size,
        center_bin_to_predict=window_size,
        position_resolution=1000,
        random_shift=100,
        random_strand=False
    )

    val_sampler.mode = "validate"
    dataloader = SamplerDataLoader(val_sampler, num_workers=val_num_workers, batch_size=val_batch_size)

    validation_sequences = []
    validation_subinputs = []
    validation_subtargets = []
    validation_targets = []

    i = 0
    for sequence, target in dataloader:  # , target_1d in dataloader:
        subinp, subtar, tar = target

        figshow(subinp[0, :, :], np=True)
        plt.show()

        validation_sequences.append(sequence)
        validation_subinputs.append(subinp)
        validation_subtargets.append(subtar)
        validation_targets.append(tar)

        i += 1
        if i == val_size:
            break

    validation_sequences = np.vstack(validation_sequences)
    validation_subinputs = np.vstack(validation_subinputs)
    validation_subtargets = np.vstack(validation_subtargets)
    validation_targets = np.vstack(validation_targets)
    del val_sampler

    print("Init train data sampler")
    out.write("Init train data sampler\n")
    sampler = RandomPositionsSampler(
        reference_sequence=MemmapGenome(
            input_path=ref_g_file_fa,
            memmapfile=ref_g_file_mmap,
            blacklist_regions="hg38",
        ),
        # target=HicSubTarFeatures(SUB_DATA_PROB, subhic_file, hic_file, normmat,
        #                           ['r1000'], [(250, 250), (250, 250)], filt_tresh=0.7,
        #                         cg_sub=True, cg_tar=True, islog_sub=True, islog_tar=True),
        target=HicAutoSubFeatures(SUBSUB_DATA_PROB, subhic_file, normmat, ['r1000'],
                                    [(250, 250), (250, 250), (250, 250)], filt_tresh=0.7,
                                    cg_sub=False, cg_tar=False, islog_sub=True, islog_tar=True),
        features=['r1000'],
        test_holdout=['chr8'],
        validation_holdout=['chr9'],
        sequence_length=window_size,
        center_bin_to_predict=window_size,
        position_resolution=1000,
        random_shift=100,
        random_strand=False
    )

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=num_workers, batch_size=batch_size, seed=seed)

    print("Start training")
    out.write("Start training\n")
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum)
    try:
        optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
        optimizer.load_state_dict(optimizer_bak)
    except:
        print("no saved optimizer found!")
    # scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.9, patience=10, threshold=0)

    i = 3000
    loss_history = []
    stime = time.time()
    for sequence, target in dataloader:
        subsubinp250, subsubtarg250, subtarg250 = target
        # subsubinp250, subtarg250 = target

        if torch.rand(1) < 0.5:
            sequence = sequence.flip([1, 2])
            subsubinp250 = subsubinp250.flip([1, 2])
            subsubtarg250 = subsubtarg250.flip([1, 2])
            subtarg250 = subtarg250.flip([1, 2])

        optimizer.zero_grad()

        seq_cuda = torch.Tensor(sequence.float()).transpose(1, 2).cuda()
        subsubinp250_cuda = subsubinp250.float().cuda()
        subsubinp250_invalid = ~torch.isfinite(subsubinp250)
        subsubinp250_cuda[subsubinp250_invalid] = 0

        ign_seq, ign_sub = ignore_sequence, ignore_subdata
        # if not ign_seq and not ign_sub:
        #    ign_sub = True if torch.rand(1) < 0.5 else False

        if ign_sub:
            targ250_cuda = subtarg250.float().cuda()
            targ250_valid = ~torch.isnan(subtarg250)
        else:
            targ250_cuda = subsubtarg250.float().cuda()
            targ250_valid = ~torch.isnan(subsubtarg250)

        pred = net(
                    seq=seq_cuda,
                    subdata=subsubinp250_cuda,
                    mask=subsubinp250_invalid,
                    ignore_seq=ign_seq,
                    ignore_sub=ign_sub
               )

        loss = (
            (
                pred[:, 0, :, :][targ250_valid] - targ250_cuda[targ250_valid]
            )
            ** 2
        ).mean()

        loss.backward()
        loss_history.append(loss.detach().cpu().numpy())
        optimizer.step()

        if i % loss_output == 0:
            print(f"{i} iteration, train loss: {np.mean(loss_history[-500:])} time elapsed {time.time() - stime}", flush=True)
            out.write(f"{i} iteration, train loss: {np.mean(loss_history[-500:])} time elapsed {time.time() - stime}\n")
            out.flush()
            stime = time.time()

        if i % validation_output == 0:
            figshow(pred[0, 0, :, :])
            plt.savefig("./png/train." + modelstr + "." + str(i) + ".pred.png")

            figshow(subtarg250[0, :, :], np=True)
            plt.savefig("./png/train." + modelstr + "." + str(i) + ".subtar.png")

            figshow(subsubinp250_cuda[0, :, :])
            plt.savefig("./png/train." + modelstr + "." + str(i) + ".subsubinp.png")

            figshow(subsubtarg250[0, :, :], np=True)
            plt.savefig("./png/train." + modelstr + "." + str(i) + ".subsubtar.png")

            if not ignore_sequence:
                torch.save(net.module.seq_encoder.state_dict(), "./models/seqpart." + modelstr + ".checkpoint")

            if not ignore_subdata:
                torch.save(net.module.hic_encoder.state_dict(), "./models/hicpart." + modelstr + ".checkpoint")

            torch.save(net.module.decoder.state_dict(), "./models/decoderpart." + out_decoder_name + "." + modelstr + ".checkpoint")

            torch.save(optimizer.state_dict(), "./models/model_" + modelstr + ".optimizer")

        if i % validation_output == 0:
            net.eval()

            corrs = []
            mse = []
            mseloss = nn.MSELoss()
            t = 0

            for sequence, subinp250, subtarg250, targ250 in zip(
                np.array_split(validation_sequences, val_size * val_batch_size),
                np.array_split(validation_subinputs, val_size * val_batch_size),
                np.array_split(validation_subtargets, val_size * val_batch_size),
                np.array_split(validation_targets, val_size * val_batch_size),
            ):
                if np.mean(np.isnan(targ250)) < 0.7:
                    subinp250_t = torch.Tensor(subinp250)
                    subinp250_cuda = subinp250_t.cuda()
                    subinp250_invalid = ~torch.isfinite(subinp250_t)
                    subinp250_cuda[subinp250_invalid] = 0

                    pred = net(
                                seq=torch.Tensor(sequence).transpose(1, 2).cuda(),
                                subdata=subinp250_cuda,
                                mask=subinp250_invalid,
                                ignore_seq=ignore_sequence,
                                ignore_sub=ignore_subdata
                            )

                    target250_t = torch.Tensor(targ250)
                    targ250_cuda = target250_t.cuda()
                    targ250_valid = ~torch.isnan(target250_t)

                    loss = (
                        (
                            pred[:, 0, :, :][targ250_valid] - targ250_cuda[targ250_valid]
                        )
                        ** 2
                    ).mean()

                    mse.append(loss.detach().cpu().numpy())

                    if t < 10:
                        figshow(pred[0, 0, :, :])
                        plt.savefig("./png/" + modelstr + ".test" + str(t) + ".pred.png")

                        figshow(subinp250[0, :, :], np=True)
                        plt.savefig("./png/" + modelstr + ".test" + str(t) + ".subinp.png")

                        figshow(targ250[0, :, :], np=True)
                        plt.savefig("./png/" + modelstr + ".test" + str(t) + ".label.png")

                    t += 1

                    pred = pred[:, 0, :, :].detach().cpu().numpy().reshape((pred.shape[0], -1))
                    target = targ250.reshape(pred.shape[0], -1)

                    for j in range(pred.shape[0]):
                        if np.mean(np.isnan(target[j, :])) < 0.7:
                            corrs.append(
                                pearsonr(
                                    pred[j, ~np.isnan(target[j, :])],
                                    target[j, ~np.isnan(target[j, :])],
                                )[0]
                            )
                        else:
                            corrs.append(np.nan)

            # scheduler.step(np.nanmean(corr_sqrt))
            # scheduler.step(np.nanmean(corr_sqrt))

            print(f"Average Corr: {np.nanmean(corrs)},  MSE {np.mean(mse)}", flush=True)
            out.write(f"Average Corr {np.nanmean(corrs)}, MSE {np.mean(mse)}\n")
            out.flush()

            del pred
            del loss
            net.train()

        if i == total_iterations:
            break

        i += 1



