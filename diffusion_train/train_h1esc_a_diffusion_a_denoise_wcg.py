import sys
import os
import re
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
import functools

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel

import selene_sdk
from selene_sdk.samplers.dataloader import SamplerDataLoader

sys.path.append("..")
from selene_utils2 import *

from preprocess_modules import HicEncoder, Decoder, DenoiseNet
from diffusion_modules import ScoreNet_a
from diffusion_utils import marginal_prob_std, loss_fn_a, diffusion_coeff, Euler_Maruyama_sampler_a

import pdb

# Suppress/hide the warning
np.seterr(invalid="ignore")

pre_modelstr = "h1esc.r1000.p1.0.wcg.noseq"
modelstr = "h1esc_a_diffusion_a_denoise_wcg"
seed = 314


torch.set_default_tensor_type("torch.FloatTensor")
os.makedirs("./models/", exist_ok=True)
os.makedirs("./png/", exist_ok=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--swa":
        print("Training in SWA mode.")
        use_swa = True
        modelstr += "_swa"
    else:
        use_swa = False

    out = open(f"log/log_{modelstr}.txt", 'w')
    
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

    print("Generate validation")
    out.write("Generate validation\n")
    
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
        
    # load preprocess module
    pre_seqPart = None
    out_decoder_name = ""
    try:
        pre_hicPart = HicEncoder(128, 2, 250)
        pre_hicPart.load_state_dict(
            torch.load("./pretrained_models/hicpart." + pre_modelstr + ".checkpoint")
        )
        print("preprocess Hic Model succefully loaded")
    except:
        print("NO SAVED HIC part of preprocess model found!")
        exit(1)
    try:
        pre_decoder = Decoder()
        pre_decoder.load_state_dict(
            torch.load("./pretrained_models/decoderpart." + out_decoder_name + "." + pre_modelstr + ".checkpoint")
        )
        print("preprocess Decoder Model succefully loaded")
    except:
        print("NO SAVED DECODER part of preprocess model found!")
        exit(1)
    
    pre_net = nn.DataParallel(DenoiseNet(seqEncoder=pre_seqPart, hicEncoder=pre_hicPart, decoder=pre_decoder))
    
    sigma = 25.0
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma)
    bceloss = nn.BCELoss()
    try:
        net = nn.DataParallel(ScoreNet_a(marginal_prob_std=marginal_prob_std_fn))
        net.load_state_dict(
            torch.load("./models/model_" + modelstr.replace("_swa", "") + ".checkpoint")
        )
    except:
        print("no saved model found!")
        net = nn.DataParallel(ScoreNet_a(marginal_prob_std=marginal_prob_std_fn))
        MODELA_PATH = "./pretrained_models/model_h1esc_a_gnorm_rcyc_1d2d.corbest.checkpoint"
        net = nn.DataParallel(ScoreNet_a(marginal_prob_std=marginal_prob_std_fn))
        net_dict = net.state_dict()
        pretrained_dict = torch.load(MODELA_PATH)
        for key in pretrained_dict.keys():
            if key in net_dict.keys():
                net_dict[key] = pretrained_dict[key]
        net.load_state_dict(net_dict)

    pre_net.cuda()
    net.cuda()
    bceloss.cuda()
    pre_net.eval()
    net.train()
    if use_swa:
        swanet = AveragedModel(net)
        swanet.train()

    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.002)
    if not use_swa:
        try:
            optimizer_bak = torch.load("./models/model_" + modelstr + ".optimizer")
            optimizer.load_state_dict(optimizer_bak)
        except:
            print("no saved optimizer found!")

    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode="max", factor=0.9, patience=10, threshold=0
    # )

    i = 0
    loss_history = []
    val_cor_best, val_mse_best, val_loss_best = 0, 1, 100000
    normmat_r = np.reshape(normmat, (250, 4, 250, 4)).mean(axis=1).mean(axis=2)
    eps = np.min(normmat_r)

    print("Init train data sampler")
    out.write("Init train data sampler\n")
    
    sampler.mode = "train"
    dataloader = SamplerDataLoader(sampler, num_workers=32, batch_size=16, seed=seed)
    
    print("Start training")
    out.write("Start training\n")
    stime = time.time()
    while True:
        for sequence, target, target_1d in dataloader:
            if torch.rand(1) < 0.5:
                sequence = sequence.flip([1, 2])
                target = target.flip([1, 2])
                target_1d = target_1d.flip([2])
            
            target_r = np.nanmean(
                np.nanmean(
                    np.reshape(target.numpy(), (target.shape[0], 250, 4, 250, 4)),
                    axis=4,
                ),
                axis=2,
            )
            # pdb.set_trace()
            target_cuda = torch.Tensor(np.log(((target_r + eps) / (normmat_r + eps)))).cuda()
            target_invalid = ~torch.isfinite(target_cuda)
            target_cuda[target_invalid] = 0
            preprocessed_target_cuda = pre_net(
                seq=None,
                subdata=target_cuda,
                mask=target_invalid,
                ignore_seq=True,
                ignore_sub=False
            )
            # preprocessed_target_cuda = preprocessed_target_cuda + torch.log(torch.tensor(2, device='cuda'))

            optimizer.zero_grad()
            loss, pred = loss_fn_a(net, torch.Tensor(sequence.float()).transpose(1, 2).cuda(), preprocessed_target_cuda, marginal_prob_std_fn)
            
            loss.backward()
            loss_history.append(loss.detach().cpu().numpy())
            optimizer.step()
           
           
            if i % 500 == 0:
                print(f"{i} iteration, train loss: {np.mean(loss_history[-500:])} time elapsed {time.time() - stime}", flush=True)
                out.write(f"{i} iteration, train loss: {np.mean(loss_history[-500:])} time elapsed {time.time() - stime}\n")
                out.flush()
                stime = time.time()
            i += 1
            if i % 500 == 0:
                figshow(pred[0, 0, :, :])
                plt.savefig("./png/model_" + modelstr + "." + str(i) + ".pred.png")
                figshow(
                    np.log(((target_r + eps) / (normmat_r + eps)))[0, :, :], np=True
                )
                plt.savefig("./png/model_" + modelstr + "." + str(i) + ".label.png")
                figshow(preprocessed_target_cuda[0, 0, :, :])
                plt.savefig("./png/model_" + modelstr + "." + str(i) + ".pre_label.png")
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
                val_loss = []
                mseloss = nn.MSELoss()
                t = 0
                for sequence, target, target_1d in zip(
                    np.array_split(validation_sequences, 256),
                    np.array_split(validation_targets, 256),
                    np.array_split(validation_target_1ds, 256),
                ):
                    target_r = np.nanmean(
                        np.nanmean(
                            np.reshape(target, (target.shape[0], 250, 4, 250, 4)),
                            axis=4,
                        ),
                        axis=2,
                    )
                    target_cuda = torch.Tensor(
                        np.log(((target_r + eps) / (normmat_r + eps)))
                    ).cuda()
                    target_invalid = ~torch.isfinite(target_cuda)
                    target_cuda[target_invalid] = 0
                    preprocessed_target_cuda = pre_net(
                        seq=None,
                        subdata=target_cuda,
                        mask=target_invalid,
                        ignore_seq=True,
                        ignore_sub=False
                    )
                    # preprocessed_target_cuda = preprocessed_target_cuda + torch.log(2)


                    loss, pred = loss_fn_a(
                        net, 
                        torch.Tensor(sequence).transpose(1, 2).cuda(), 
                        preprocessed_target_cuda, 
                        marginal_prob_std_fn
                    )
                    val_loss.append(loss.detach().cpu().numpy())
                    
                    if t == 0:
                        pred = Euler_Maruyama_sampler_a(net,
                                                        torch.Tensor(sequence).transpose(1, 2).cuda(),
                                                        250, 
                                                        marginal_prob_std_fn,
                                                        diffusion_coeff_fn,
                                                        sequence.shape[0],
                                                        device='cuda')
                    
                        for idx in range(sequence.shape[0]):
                            figshow(pred[idx, 0, :, :])
                            plt.savefig(
                                "./png/model_" + modelstr + ".test" + str(idx) + ".pred.png"
                            )
                            figshow(
                                np.log(((target_r + eps) / (normmat_r + eps)))[idx, :, :],
                                np=True,
                            )
                            plt.savefig(
                                "./png/model_" + modelstr + ".test" + str(idx) + ".label.png"
                            )
                            figshow(preprocessed_target_cuda[idx, 0, :, :])
                            plt.savefig(
                                "./png/model_" + modelstr + ".test" + str(idx) + ".pre_label.png"
                            )            

                    t += 1
                        # if np.mean(np.isnan(target_r)) < 0.7:
                        #     target_cuda = torch.Tensor(
                        #         np.log(((target_r + eps) / (normmat_r + eps)))
                        #     ).cuda()
                        #     loss = (
                        #         (
                        #             pred[:, 0, :, :][~torch.isnan(target_cuda)]
                        #             - target_cuda[~torch.isnan(target_cuda)]
                        #         )
                        #         ** 2
                        #     ).mean()
                        #     mse.append(loss.detach().cpu().numpy())
                        #     pred = (
                        #         pred[:, 0, :, :]
                        #         .detach()
                        #         .cpu()
                        #         .numpy()
                        #         .reshape((pred.shape[0], -1))
                        #     )
                        #     target = np.log(((target_r + eps) / (normmat_r + eps))).reshape(
                        #         (pred.shape[0], -1)
                        #     )
                        #     for j in range(pred.shape[0]):
                        #         if np.mean(np.isnan(target[j, :])) < 0.7:
                        #             corr.append(
                        #                 pearsonr(
                        #                     pred[j, ~np.isnan(target[j, :])],
                        #                     target[j, ~np.isnan(target[j, :])],
                        #                 )[0]
                        #             )
                        #         else:
                        #             corr.append(np.nan)
                    # corr_mean = np.nanmean(corr)
                    # mse_mean = np.mean(mse)
                    # scheduler.step(corr_mean)
                    # if val_cor_best < corr_mean:
                    #     torch.save(
                    #         net.state_dict(),
                    #         "./models/model_" + modelstr + ".corbest.checkpoint",
                    #     )
                    #     val_cor_best = corr_mean
                    # if val_mse_best > mse_mean:
                    #     torch.save(
                    #         net.state_dict(),
                    #         "./models/model_" + modelstr + ".msebest.checkpoint",
                    #     )
                    #     val_mse_best = mse_mean
                val_loss_mean = np.mean(val_loss)
                if val_loss_best > val_loss_mean:
                    torch.save(
                        net.state_dict(),
                        "./models/model_" + modelstr + ".lossbest.checkpoint",
                    )
                    val_loss_best = val_loss_mean
                print(
                    "Average Loss{0}".format(
                        val_loss_mean
                    )
                )
                out.write(
                    "Average Loss{0}".format(
                        val_loss_mean
                    )
                )
                del pred
                del loss
                if use_swa:
                    swanet.train()
                else:
                    net.train()
