#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import math
import numpy as np
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="mean pools representations by compressing uniform splits of the data"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--subsample-rate', type=float, default=0.5, help='size to subsample data to')

    parser.add_argument('--remove-extra', action='store_true', help='if true, removes extra states that cant be pooled, otherwise pads with 0s')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train

    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r") # [1626, 16], /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.npy, NOTE read

    os.makedirs(args.save_dir, exist_ok=True) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled, NOTE write to path
    save_path = osp.join(args.save_dir, args.split) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled/train

    copyfile(source_path + ".tsv", save_path + ".tsv") # NOTE copy directly, from /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.tsv to /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled/train.tsv

    if os.path.exists(source_path + ".phn"): # not in
        copyfile(source_path + ".phn", save_path + ".phn")
    if os.path.exists(source_path + ".wrd"): # not in
        copyfile(source_path + ".wrd", save_path + ".wrd")

    if os.path.exists(osp.join(args.source, "dict.phn.txt")): # not in
        copyfile(
            osp.join(args.source, "dict.phn.txt"),
            osp.join(args.save_dir, "dict.phn.txt"),
        )

    if osp.exists(save_path + ".npy"): # not in
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy") # NOTE write to, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled/train.npy

    with open(source_path + ".lengths", "r") as lf: # NOTE read, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.lengths
        lengths = lf.readlines() # e.g., merged (cluster.id after k-means) 1 1 1 to 1
        # ['122\n', '187\n', '146\n', '289\n', '62\n', '457\n', '363\n']

    fsz = features.shape[-1] # hidden dimension size = 16
    start = 0
    with torch.no_grad():
        with open(save_path + ".lengths", "w") as lengths_out: # NOTE write to, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled/train.lengths
            for length in tqdm.tqdm(lengths):
                length = int(length)
                end = start + length
                feats = features[start:end] # e.g., feats.shape=[122, 16]
                start += length
                x = torch.from_numpy(feats).cuda()
                target_num = math.ceil(length * args.subsample_rate) # 122*0.5=61=target_num
                rem = length % target_num

                if rem > 0:
                    if args.remove_extra:
                        to_rem = target_num - rem
                        target_num -= 1
                        x = x[:-to_rem]
                    else:
                        to_add = target_num - rem
                        x = F.pad(x, [0, 0, 0, to_add])
                        x[-to_add:] = x[-to_add - 1]

                x = x.view(target_num, -1, fsz) # from [122, 16] to [61, 2, 16]
                x = x.mean(dim=-2) # from [61, 2, 16] to [61, 16]
                print(target_num, file=lengths_out)
                npaa.append(x.cpu().numpy()) # before [1626, 16] -> after, [815, 16] 


if __name__ == "__main__":
    main()


    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.npy
    #         train.npy
    #         train.lengths
    #         train.tsv - copied directly


    # outputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled
    #          train.tsv - copied directly
    #          train.npy
    #          train.lengths
    #      
    #          valid.lengths, valid.npy, valid.tsv
