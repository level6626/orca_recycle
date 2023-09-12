import sys
import os
import re

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.swa_utils import AveragedModel

import selene_sdk
from selene_sdk.samplers.dataloader import SamplerDataLoader

sys.path.append("..")
from selene_utils2 import *
from recycle_modules import NetRcyc1d2d, Netf

import pdb

# Suppress/hide the warning
np.seterr(invalid="ignore")

modelstr = "h1esc_a_rcyc_1d2d_2stage"
seed = 314
N_rcyc = 2


torch.set_default_tensor_type("torch.FloatTensor")
os.makedirs("./models/", exist_ok=True)
os.makedirs("./png/", exist_ok=True)

MODELA_PATH = "./models/orca_h1esc.net0.statedict"
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--swa":
        print("Training in SWA mode.")
        use_swa = True
        modelstr += "_swa"
    else:
        use_swa = False

    normmat_bydist = np.exp(
        np.load("../resources/4DNFI9GMP2J8.rebinned.mcool.expected.res1000.npy")
    )[:1000]
    normmat = normmat_bydist[
        np.abs(np.arange(1000)[:, None] - np.arange(1000)[None, :])
    ]

    t = Genomic2DFeatures(
        ["../resources/4DNFI9GMP2J8.rebinned.mcool::/resolutions/1000"],
        ["r1000"],
        (1000, 1000),
        cg=True,
    )
    sampler = RandomPositionsSamplerHiC(
        reference_sequence=MemmapGenome(
            input_path="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa",
            memmapfile="../resources/Homo_sapiens.GRCh38.dna.primary_assembly.fa.mmap",
            blacklist_regions="hg38",
        ),
        target=t,
        target_1d=MultibinGenomicFeatures(
            "../resources/h1esc/h1esc.hg38.bed.sorted.gz",
            np.loadtxt("../resources/h1esc/h1esc.hg38.bed.sorted.features", str),
            4000,
            4000,
            (32, 250),
            mode="any",
        ),
        features=["r1000"],
        test_holdout=["chr9", "chr10"],
        validation_holdout=["chr8"],
        sequence_length=1000000,
        position_resolution=1000,
        random_shift=100,
        random_strand=False,
        cross_chromosome=False,
    )

    sampler.mode = "validate"
    dataloader = SamplerDataLoader(sampler, num_workers=32, batch_size=16)

    validation_sequences = []
    validation_targets = []
    validation_target_1ds = []

    i = 0
    for sequence, target, target_1d in dataloader:
        validation_sequences.append(sequence)
        validation_targets.append(target)
        validation_target_1ds.append(target_1d)
        i += 1
        if i == 128:
            break

    validation_sequences = np.vstack(validation_sequences)
    validation_targets = np.vstack(validation_targets)
    validation_target_1ds = np.vstack(validation_target_1ds)

    def figshow(x, np=False):
        if np:
            plt.imshow(x.squeeze())
        else:
            plt.imshow(x.squeeze().cpu().detach().numpy())
        plt.show()

    bceloss = nn.BCELoss()
    try:
        net = nn.DataParallel(NetRcyc1d2d(num_1d=32))
        net.load_state_dict(
            torch.load("./models/model_" + modelstr.replace("_swa", "") + ".checkpoint")
        )
    except:
        print("no saved model found!")
        net = nn.DataParallel(NetRcyc1d2d(num_1d=32))

    net0 = nn.DataParallel(Netf(num_1d=32))
    net0_dict = net0.state_dict()
    pretrained_dict = torch.load(MODELA_PATH)
    for key in list(pretrained_dict.keys())[1:]:
            net0_dict[re.sub(r"^module\.", "", key)] = pretrained_dict[key]
    net0.load_state_dict(net0_dict)
    for param in net0.parameters():
        param.requires_grad = False
    
    net0.cuda()
    net.cuda()
    bceloss.cuda()
    net0.eval()
    net.train()
    if use_swa:
        swanet = AveragedModel(net)
        swanet.train()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.002, momentum=0.98)
    if not use_swa:
        try:
            optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
            optimizer.load_state_dict(optimizer_bak)
        except:
            print("no saved optimizer found!")

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.9, patience=10, threshold=0
    )
    ## break
    i = 1999
    loss_history = []
    val_cor_best, val_mse_best = 0, 1
    normmat_r = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
    eps = np.min(normmat_r)
    
    def logit(p):
        return torch.log(p / (1 - p))

    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=32, batch_size=16, seed=seed)
    while True:
        for sequence, target, target_1d in dataloader:
            if torch.rand(1) < 0.5:
                sequence = sequence.flip([1, 2])
                target = target.flip([1, 2])
                target_1d = target_1d.flip([2])

            optimizer.zero_grad()
            
            pre_pred, pre_pred_1d = net0(
                    torch.Tensor(sequence.float()).transpose(1, 2).cuda())
            pred, pred_1d = net(
                torch.Tensor(sequence.float()).transpose(1, 2).cuda(), pre_pred
            )
            pred = pre_pred + pred
            pred_1d = torch.sigmoid(logit(pre_pred_1d) + logit(pred_1d))
            
            target_r = np.nanmean(
                np.nanmean(
                    np.reshape(target.numpy(), (target.shape[0], 250, 4, 250, 4)),
                    axis=4,
                ),
                axis=2,
            )

            target_cuda = torch.Tensor(
                np.log(((target_r + eps) / (normmat_r + eps)))
            ).cuda()
            
            loss = (
                (
                    pred[:, 0, :, :][~torch.isnan(target_cuda)]
                    - target_cuda[~torch.isnan(target_cuda)]
                )
                ** 2
            ).mean()
            loss_1d = bceloss(pred_1d, torch.Tensor(target_1d.float()).cuda())
            
            loss = loss + loss_1d
            loss.backward()
            loss_history.append(loss.detach().cpu().numpy())
            optimizer.step()
            if use_swa:
                swanet.update_parameters(net)
                with torch.no_grad():
                    swanet(torch.Tensor(sequence.float()).transpose(1, 2).cuda())

            if i % 500 == 0:
                print("Train loss: ", np.mean(loss_history[-500:]), flush=True)
            i += 1
            if i % 500 == 0:
                figshow(pred[0, 0, :, :])
                plt.savefig("./png/model_" + modelstr + "." + str(i) + ".pred.png")
                figshow(
                    np.log(((target_r + eps) / (normmat_r + eps)))[0, :, :], np=True
                )
                plt.savefig("./png/model_" + modelstr + "." + str(i) + ".label.png")
                if use_swa:
                    torch.save(
                        swanet.module.state_dict(),
                        "./models/model_" + modelstr + ".checkpoint",
                    )
                else:
                    torch.save(
                        net.state_dict(), "./models/model_" + modelstr + ".checkpoint"
                    )
                    torch.save(
                        optimizer.state_dict(),
                        "./models/model_" + modelstr + ".optimizer",
                    )

            if i % 2000 == 0:
                if use_swa:
                    swanet.eval()
                else:
                    net.eval()

                corr = []
                mse = []
                loss_1ds = []
                mseloss = nn.MSELoss()
                t = 0
                for sequence, target, target_1d in zip(
                    np.array_split(validation_sequences, 256),
                    np.array_split(validation_targets, 256),
                    np.array_split(validation_target_1ds, 256),
                ):
                    pre_pred, pre_pred_1d = net0(
                        torch.Tensor(sequence).transpose(1, 2).cuda())
                    pred, pred_1d = net(
                        torch.Tensor(sequence).transpose(1, 2).cuda(), pre_pred
                    )
                    pred = pre_pred + pred
                    pred_1d = torch.sigmoid(logit(pre_pred_1d) + logit(pred_1d))

                    target_r = np.nanmean(
                        np.nanmean(
                            np.reshape(target, (target.shape[0], 250, 4, 250, 4)),
                            axis=4,
                        ),
                        axis=2,
                    )
                    if t < 10:
                        figshow(pred[0, 0, :, :])
                        plt.savefig(
                            "./png/model_" + modelstr + ".test" + str(t) + ".pred.png"
                        )
                        figshow(
                            np.log(((target_r + eps) / (normmat_r + eps)))[0, :, :],
                            np=True,
                        )
                        plt.savefig(
                            "./png/model_" + modelstr + ".test" + str(t) + ".label.png"
                        )
                    t += 1
                    if np.mean(np.isnan(target_r)) < 0.7:
                        loss_1d = bceloss(
                            pred_1d, torch.Tensor(target_1d).float().cuda()
                        )
                        loss_1ds.append(loss_1d.detach().cpu().numpy())
                        target_cuda = torch.Tensor(
                            np.log(((target_r + eps) / (normmat_r + eps)))
                        ).cuda()
                        loss = (
                            (
                                pred[:, 0, :, :][~torch.isnan(target_cuda)]
                                - target_cuda[~torch.isnan(target_cuda)]
                            )
                            ** 2
                        ).mean()
                        mse.append(loss.detach().cpu().numpy())
                        pred = (
                            pred[:, 0, :, :]
                            .detach()
                            .cpu()
                            .numpy()
                            .reshape((pred.shape[0], -1))
                        )
                        target = np.log(((target_r + eps) / (normmat_r + eps))).reshape(
                            (pred.shape[0], -1)
                        )
                        for j in range(pred.shape[0]):
                            if np.mean(np.isnan(target[j, :])) < 0.7:
                                corr.append(
                                    pearsonr(
                                        pred[j, ~np.isnan(target[j, :])],
                                        target[j, ~np.isnan(target[j, :])],
                                    )[0]
                                )
                            else:
                                corr.append(np.nan)
                corr_mean = np.nanmean(corr)
                mse_mean = np.mean(mse)
                scheduler.step(corr_mean)
                if val_cor_best < corr_mean:
                    torch.save(
                        net.state_dict(),
                        "./models/model_" + modelstr + ".corbest.checkpoint",
                    )
                    val_cor_best = corr_mean
                if val_mse_best > mse_mean:
                    torch.save(
                        net.state_dict(),
                        "./models/model_" + modelstr + ".msebest.checkpoint",
                    )
                    val_mse_best = mse_mean
                print(
                    "Average Corr{0}, MSE {1}, 1D {2}".format(
                        corr_mean, mse_mean, np.mean(loss_1ds)
                    )
                )
                del pred
                del loss
                if use_swa:
                    swanet.train()
                else:
                    net.train()
