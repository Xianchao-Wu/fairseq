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
import sys

import faiss
import torch.nn.functional as F

from wav2vec_cluster_faiss import parse_faiss_specs, Wav2VecFeatureReader


def get_parser():
    parser = argparse.ArgumentParser(description="apply clusters")
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='split to process', required=True)
    parser.add_argument('--labels', help='split to process', default="phn")
    parser.add_argument('--path', help='path to pca and centroids', required=True)
    parser.add_argument('--checkpoint', type=str, 
            help='checkpoint for wav2vec model (if using wav2vec features)', required=True)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=14) # NOTE TODO
    parser.add_argument('--max-tsz', type=int, help='batch kmeans up to this much', default=14)
    # fmt: on

    return parser


def get_iterator(args):
    label_path = osp.join(args.data, f"{args.split}.{args.labels}") 
    # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.phn', 
    # not prepared yet... NOTE

    if osp.exists(label_path):
        lp = open(label_path, "r")
    else:
        lp = None # Here

    with open(osp.join(args.data, f"{args.split}.tsv"), "r") as fp: 
        # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv (read)
        lines = fp.read().split("\n")
        root = lines.pop(0).strip() 
        # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads'

        files = [line.rstrip() for line in lines if len(line) > 0]
        # files=['test-clean-wav-debug/908-31957-0005.wav\t54240', ...]
        if lp is not None:
            lbls = [line.rstrip() for line in lp]
        else:
            lbls = [None] * len(files) # [None, None, None, None, None, None, None], here!

        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer) # NOTE 注意，这里还是导入checkpoint!!!
        # ('/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt', 11)
        def iterate():
            #import ipdb; ipdb.set_trace()
            for fname, lbl in zip(files, lbls):
                file = osp.join(root, fname.split("\t")[0]) 
                # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads/test-clean-wav-debug/908-31957-0005.wav

                feats = reader.get_feats(file) # torch.Size([169, 768])
                yield feats.data, fname, lbl
                # 1. feats.data.shape=[169, 768], 
                # 2. fname=test-clean-wav-debug/908-31957-0005.wav\t54240, 
                # 3. lbl=None
        return iterate, num, root


def main():
    parser = get_parser()
    args = parser.parse_args()

    spec = osp.basename(args.path) # 'CLUS128'

    try:
        faiss_spec = parse_faiss_specs(spec.rstrip("/"))[0] 
        # faiss_spec(pca=0, norm=False, n_clus=128, sphere=False, spec_str='CLUS128')
    except:
        print(spec)
        raise

    print("Faiss Spec:", faiss_spec, file=sys.stderr)

    if faiss_spec.pca: # 0, not in
        A = torch.from_numpy(np.load(osp.join(args.path, "pca_A.npy"))).cuda()
        b = torch.from_numpy(np.load(osp.join(args.path, "pca_b.npy"))).cuda()
        print("Loaded PCA", file=sys.stderr)

    centroids = np.load(osp.join(args.path, "centroids.npy")) # NOTE read centroids.npy
    # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/centroids.npy', centroids.shape=(128, 768) NOTE
    print("Loaded centroids", centroids.shape, file=sys.stderr)

    res = faiss.StandardGpuResources() 
    # <faiss.swigfaiss_avx2.StandardGpuResources; proxy of <Swig Object of type 'faiss::gpu::StandardGpuResources *' at 0x7fb3785c8600> >
    index_flat = (
        faiss.IndexFlatL2(centroids.shape[1])
        if not faiss_spec.sphere
        else faiss.IndexFlatIP(centroids.shape[1])
    ) 
    # <faiss.swigfaiss_avx2.IndexFlatL2; proxy of <Swig Object of type 'faiss::IndexFlatL2 *' at 0x7fb231885ed0> >

    faiss_index = faiss.index_cpu_to_gpu(res, 0, index_flat) # NOTE
    faiss_index.add(centroids)
    #import ipdb; ipdb.set_trace()
    generator, num, root = get_iterator(args)
    iterator = generator()

    had_labels = False
    label_path = osp.join(args.path, f"{args.split}.{args.labels}")
    # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.phn'
    with torch.no_grad():
        with open(osp.join(args.path, f"{args.split}.src"), "w") as fp, open(
            osp.join(args.path, f"{args.split}.tsv"), "w"
        ) as pp, open(label_path, "w") as lp:
            # 1. fp, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.src, to write to, cluster ids
            # 2. pp, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.tsv, wave file path
            # 3. lp, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/train.phn, label path
            print(root, file=pp) # root = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads
            for f, fname, lbl in tqdm.tqdm(iterator, total=num):
                #import ipdb; ipdb.set_trace()
                if faiss_spec.pca: # 0, not in!
                    f = torch.mm(f, A) + b
                if faiss_spec.norm: # False, not in!
                    f = F.normalize(f, p=2, dim=-1)

                f = f.cpu().numpy() # one file, f.shape=[169, 768]

                _, z = faiss_index.search(f, 1) 
                # NOTE important f.shape=[169, 768], z.shape=[169, 1]; z=[[107], [107], ...]

                print(" ".join(str(x.item()) for x in z), file=fp) # e.g., '107 107 18 107 ...'
                print(fname, file=pp) # fname='test-clean-wav-debug/908-31957-0005.wav\t54240', to train.tsv

                if lbl is not None: # None, not in
                    print(lbl, file=lp)
                    had_labels = True
    if not had_labels:
        os.remove(label_path)


if __name__ == "__main__":
    main()

    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv, valid.tsv
    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/centroids.npy

    # outputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/
    #          train.src, train.tsv
    #          valid.src, valid.tsv

