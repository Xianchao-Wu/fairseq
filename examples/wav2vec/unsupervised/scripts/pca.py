#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import numpy as np

import faiss
#import mkl
#mkl.get_max_threads()


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute a pca matrix given an array of numpy features"
    )
    # fmt: off
    parser.add_argument('data', help='numpy file containing features')
    parser.add_argument('--output', help='where to save the pca matrix', required=True)
    parser.add_argument('--dim', type=int, help='dim for pca reduction', required=True)
    parser.add_argument('--eigen-power', type=float, default=0, help='eigen power, -0.5 for whitening')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    print("Reading features")
    x = np.load(args.data, mmap_mode="r") # e.g., [2196, 32], combined.seq.len=2196, dim=32
    # args.data='/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.npy' NOTE, read    print("Computing PCA")
    pca = faiss.PCAMatrix(x.shape[-1], args.dim, args.eigen_power) # TODO bad thing, x.shape[-1]=32, args.dim=512, and 32 < 512, so fail. Solution: should set x.shape[-1]=768! -> <faiss.swigfaiss_avx2.PCAMatrix; proxy of <Swig Object of type 'faiss::PCAMatrix *' at 0x7fefbf929a80> >
    pca.train(x)
    b = faiss.vector_to_array(pca.b) # bias=(16,)
    A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in) # pca.d_out=16, pca.d_in=32; A.shape=(16, 32)

    os.makedirs(args.output, exist_ok=True) # args.output='/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/pca'

    prefix = str(args.dim) # '16'
    if args.eigen_power != 0: # 0, not in
        prefix += f"_{args.eigen_power}"

    np.save(osp.join(args.output, f"{prefix}_pca_A"), A.T) # A.T.shape=(32, 16)
    np.save(osp.join(args.output, f"{prefix}_pca_b"), b) # (16,)


if __name__ == "__main__":
    main()


    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.npy

    # outputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/pca
    #          16_pca_A.npy, 16_pca_b.npy
