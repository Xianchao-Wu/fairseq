#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np
import tqdm
import torch
import random
from shutil import copyfile

from npy_append_array import NpyAppendArray


def get_parser():
    parser = argparse.ArgumentParser(
        description="transforms features via a given pca and stored them in target dir"
    )
    # fmt: off
    parser.add_argument('source', help='directory with features')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--cluster-dir', help='where the clusters are')
    parser.add_argument('--pooling', type=str, default='mean', choices=['mean', 'sample'], help='how to pool')
    # fmt: on

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    source_path = osp.join(args.source, args.split) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train
    cluster_path = osp.join(args.cluster_dir, args.split + ".src") # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.src
    print(f"data path: {source_path}")

    features = np.load(source_path + ".npy", mmap_mode="r") # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.npy, NOTE, read, e.g., [2196, 16] this is after pca (from 32dim to 16dim)
    sizes = []
    offsets = []
    offset = 0
    with open(source_path + ".lengths", "r") as len_f: # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.lengths, NOTE, read
        for line in len_f:
            length = int(line.rstrip()) # e.g., '169\n'
            sizes.append(length)
            offsets.append(offset)
            offset += length

    clusters = []
    with open(cluster_path, "r") as cf: # e.g., 'CLUS128/train.src' /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.src, NOTE, read
        for line in cf:
            line = line.rstrip() # e.g., 22 22 57 22 22 57 86 ... (cluster IDs of each re-scaled "frame")
            items = line.split()
            items = list(map(int, items))
            clusters.append(items)

    os.makedirs(args.save_dir, exist_ok=True) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean, NOTE, write output path
    save_path = osp.join(args.save_dir, args.split) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train

    copyfile(source_path + ".tsv", save_path + ".tsv") # copy from /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.tsv to /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.tsv

    if os.path.exists(source_path + ".phn"): # not in
        copyfile(source_path + ".phn", save_path + ".phn")
    if os.path.exists(osp.join(args.source, "dict.phn.txt")): # not in
        copyfile(
            osp.join(args.source, "dict.phn.txt"),
            osp.join(args.save_dir, "dict.phn.txt"),
        )
    if os.path.exists(source_path + ".wrd"): # not in
        copyfile(source_path + ".wrd", save_path + ".wrd")

    if osp.exists(save_path + ".npy"):
        os.remove(save_path + ".npy")
    npaa = NpyAppendArray(save_path + ".npy") # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.npy, NOTE write output file

    def merge(feats, clust):
        #import ipdb; ipdb.set_trace()
        feats = torch.from_numpy(feats.copy()) # [250, 16]
        clust = torch.LongTensor(clust) # [250]
        _, counts = clust.unique_consecutive(return_counts=True) # e.g., 1 1 1 -> 1 with duplication count=3; (250)->(187)
        curr = 0

        merged = []
        for c in counts:
            c = c.item()
            start = curr
            end = curr + c
            curr += c
            if args.pooling == "mean":
                new_x = feats[start:end].mean(dim=0) 
                # here, NOTE average the vectors to obtain a new vector representation
            elif args.pooling == "sample":
                new_x = feats[start + int(random.random() * c)]
            else:
                raise NotImplementedError()
            merged.append(new_x)

        return torch.stack(merged, dim=0).numpy()

    with open(save_path + ".lengths", "w") as l_f: # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.lengths, NOTE, write file
        for size, offset, clust in tqdm.tqdm(
            zip(sizes, offsets, clusters), total=len(sizes)
        ):
            end = size + offset
            feats = features[offset:end]
            feats = merge(feats, clust)
            print(len(feats), file=l_f)
            npaa.append(feats) # (122, 16) + ... -> 


if __name__ == "__main__":
    main()


    # inputs:
    # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.npy, NOTE, read, e.g., [2196, 16] this is after pca (from 32dim to 16dim)
    # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.lengths, NOTE, read
    # e.g., 'CLUS128/train.src' /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.src, NOTE, read


    # outputs:
    #os.makedirs(args.save_dir, exist_ok=True) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean, NOTE, write output path
    #npaa = NpyAppendArray(save_path + ".npy") # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.npy, NOTE write output file
    #with open(save_path + ".lengths", "w") as l_f: # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.lengths, NOTE, write file
    # copy from /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16/train.tsv to /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean/train.tsv

    # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean
    # train.lengths, 122, 177, ..., 348
    # train.npy [1626, 16]
    # train.tsv [copied directly]
   
    # valid.lengths, valid.npy, valid.tsv 
