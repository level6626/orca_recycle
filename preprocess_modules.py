import cooler
from cooltools.lib.numutils import adaptive_coarsegrain
import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

import selene_sdk
from selene_sdk.targets import Target


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


class DenoiseNet(nn.Module):
    def __init__(self, seqEncoder, hicEncoder, decoder):
        super(DenoiseNet, self).__init__()
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