# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import numpy as np

import joblib
import torch
import tqdm

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_km_label")


class ApplyKmeans(object):
    def __init__(self, km_path):
        self.km_model = joblib.load(km_path) # NOTE read, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)

        self.C = torch.from_numpy(self.C_np)
        self.Cnorm = torch.from_numpy(self.Cnorm_np)
        if torch.cuda.is_available():
            self.C = self.C.cuda()
            self.Cnorm = self.Cnorm.cuda()

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            dist = (
                x.pow(2).sum(1, keepdim=True)
                - 2 * torch.matmul(x, self.C)
                + self.Cnorm
            )
            return dist.argmin(dim=1).cpu().numpy()
        else:
            dist = (
                (x ** 2).sum(1, keepdims=True)
                - 2 * np.matmul(x, self.C_np)
                + self.Cnorm_np
            )
            return np.argmin(dist, axis=1)


def get_feat_iterator(feat_dir, split, nshard, rank):
    feat_path = f"{feat_dir}/{split}_{rank}_{nshard}.npy" # NOTE read /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/train_0_1.npy
    leng_path = f"{feat_dir}/{split}_{rank}_{nshard}.len" # NOTE read /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/train_0_1.len
    with open(leng_path, "r") as f:
        lengs = [int(line.rstrip()) for line in f]
        offsets = [0] + np.cumsum(lengs[:-1]).tolist()

    def iterate():
        feat = np.load(feat_path, mmap_mode="r")
        assert feat.shape[0] == (offsets[-1] + lengs[-1])
        for offset, leng in zip(offsets, lengs):
            yield feat[offset: offset + leng]

    return iterate, len(lengs)


def dump_label(feat_dir, split, km_path, nshard, rank, lab_dir):
    # feat_dir='/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc'
    # split='train', 
    # km_path=/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64
    # nshard=1, rank=0
    # lab_dir='/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64_idx'

    apply_kmeans = ApplyKmeans(km_path) # 初始化一个对象, ApplyKmeans
    generator, num = get_feat_iterator(feat_dir, split, nshard, rank)
    iterator = generator()

    lab_path = f"{lab_dir}/{split}_{rank}_{nshard}.km" # NOTE write to file, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64_idx/train_0_1.km
    os.makedirs(lab_dir, exist_ok=True)
    with open(lab_path, "w") as f:
        for feat in tqdm.tqdm(iterator, total=num):
            # feat = torch.from_numpy(feat).cuda()
            lab = apply_kmeans(feat).tolist()
            f.write(" ".join(map(str, lab)) + "\n")
    logger.info("finished successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feat_dir")
    parser.add_argument("split")
    parser.add_argument("km_path")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("lab_dir")
    args = parser.parse_args()
    logging.info(str(args))
    #import ipdb; ipdb.set_trace()
    dump_label(**vars(args))

    # input: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64 (one file) 
    # input: cls64, train_0_1.len, train_0_1.npy 

    # output: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64_idx/train_0_1.km
