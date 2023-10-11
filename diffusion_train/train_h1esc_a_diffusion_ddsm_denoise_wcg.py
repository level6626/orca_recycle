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
from diffusion_utils import marginal_prob_std, loss_fn_unc, diffusion_coeff, Euler_Maruyama_sampler_unc

import pdb

# Suppress/hide the warning
np.seterr(invalid="ignore")

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""
  def __init__(self, embed_dim, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, padding=4)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, output_padding=1)
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1)
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1, padding=4)

    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std

  def forward(self, x, t):
    # Obtain the Gaussian random feature embedding for t
    embed = self.act(self.embed(t))
    # Encoding path
    h1 = self.conv1(x)
    ## Incorporate information from t
    h1 += self.dense1(embed)
    ## Group normalization
    h1 = self.gnorm1(h1)
    h1 = self.act(h1)
    h2 = self.conv2(h1)
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    h3 = self.conv3(h2)
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)
    h4 = self.conv4(h3)
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    
    # Decoding path
    h = self.tconv4(h4)
    ## Skip connection from the encoding path
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    h = self.tconv3(torch.cat([h, h3], dim=1))
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    h = self.tconv2(torch.cat([h, h2], dim=1))
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    # h = h / self.marginal_prob_std(t)[:, None, None, None]
    return h


pre_modelstr = "h1esc.r1000.p1.0.wcg.noseq"
modelstr = "h1esc_a_diffusion_ddsm_denoise_wcg"
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
        net = nn.DataParallel(ScoreNet(marginal_prob_std_fn))
        net.load_state_dict(
            torch.load("./models/" + modelstr + ".checkpoint")
        )
        print("Saved model successfully loaded.")
    except:
        net = nn.DataParallel(ScoreNet(marginal_prob_std_fn))
        print("NO SAVED model found!")
    
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

            optimizer.zero_grad()
            loss = loss_fn_unc(net, preprocessed_target_cuda, marginal_prob_std_fn)
            
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
                pred = Euler_Maruyama_sampler_unc(net,
                                                250, 
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn,
                                                1,
                                                device='cuda')
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

                    loss = loss_fn_unc(
                        net, 
                        preprocessed_target_cuda, 
                        marginal_prob_std_fn
                    )
                    val_loss.append(loss.detach().cpu().numpy())
                    
                    if t == 0:
                        pred = Euler_Maruyama_sampler_unc(net,
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
