#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import gc
import os
import os.path as osp
import random
import numpy as np
import tqdm
import torch

from collections import namedtuple

import faiss
import sys
sys.path.append('/workspace/asr/wav2vec/fairseq')
import fairseq
import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec model (if using wav2vec features)', required=True)
    parser.add_argument('--sample-pct', '-r', type=float, help='percentage of timesteps to sample', default=0)
    parser.add_argument('--layer', '-l', type=int, help='which layer to read', default=11) # NOTE default was 14, but small.ckpt only has 12 layers...
    parser.add_argument('--faiss-specs', '-f', type=str,
                        help='faiss index specs; separated by space '
                             'format is: PCAx_NORM_CLUSx_SPHERICAL -> '
                                'PCAx if exists first apply PCA '
                                'NORM if exists, normalize the vector by L2 norm '
                                'CLUSx must exist, cluster to x clusters '
                                'SPEHRICAL if exists, apply spherical kmeans',
                        default='l2')
    # fmt: on

    return parser


faiss_spec = namedtuple("faiss_spec", ["pca", "norm", "n_clus", "sphere", "spec_str"])


def parse_faiss_specs(specs_str):
    specs = []
    for ss in specs_str.split():
        comps = ss.split("_")
        pca = 0
        norm = False
        n_clus = 0
        sphere = False
        for c in comps:
            if c.startswith("PCA"):
                pca = int(c[3:])
            elif c == "NORM":
                norm = True
            elif c.startswith("CLUS"):
                n_clus = int(c[4:])
            elif c == "SPHERICAL":
                sphere = True
        assert n_clus > 0
        specs.append(
            faiss_spec(pca=pca, norm=norm, n_clus=n_clus, sphere=sphere, spec_str=ss)
        )
    return specs


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(cp_file) # NOTE 读取checkpoint!
        # state.keys=dict_keys(['args', 'model', 'optimizer_history', 'extra_state', 'last_optimizer_state', 'cfg'])
        self.layer = layer # 11

        if "cfg" in state:
            w2v_args = state["cfg"] # 一大坨配置信息。。。
            # w2v_args.task=ipdb> p w2v_args.task
            #{'_name': 'audio_pretraining', 'data': '/checkpoint/abaevski/data/speech/libri/100h/wav2vec/raw/', 'labels': 'ltr', 'binarized_dataset': False, 'sample_rate': 16000, 'normalize': False, 'enable_padding': False, 'max_sample_size': None, 'min_sample_size': None, 'num_batch_buckets': 0, 'precompute_mask_indices': False, 'inferred_w2v_config': None, 'tpu': True, 'text_compression_level': 'none'}

            task = fairseq.tasks.setup_task(w2v_args.task) # <fairseq.tasks.audio_pretraining.AudioPretrainingTask object at 0x7fded5040340>
            model = task.build_model(w2v_args.model)
        else:
            w2v_args = state["args"]
            task = fairseq.tasks.setup_task(w2v_args)
            model = task.build_model(w2v_args)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        model.cuda()
        self.model = model

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            source = torch.from_numpy(x).view(1, -1).float().cuda()
            res = self.model(
                source=source, padding_mask=None, mask=False, features_only=True, layer=self.layer
            )
            return res["layer_results"][self.layer][0].squeeze(1)


def get_iterator(args):
    with open(args.data, "r") as fp: # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv'
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        # files = ['/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads/test-clean-wav-debug/908-31957-0005.wav', ...]
        if getattr(args, "sample_pct", 0) > 0: # 1.0
            files = random.sample(files, int(args.sample_pct * len(files)))
        num = len(files) # still 7
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer)
        # ('/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt', 11)
        def iterate():
            for fname in files:
                feats = reader.get_feats(fname) # fname="'/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads/test-clean-wav-debug/908-31957-0005.wav'", feats.shape=torch.Size([169, 768]) NOTE not 32 anymore!
                yield feats.cpu().numpy()

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    faiss_specs = parse_faiss_specs(args.faiss_specs)
    print("Faiss Specs:", faiss_specs)
    # Faiss Specs: [faiss_spec(pca=0, norm=False, n_clus=128, sphere=False, spec_str='CLUS128')]
    feat_path = osp.join(args.save_dir, "features") # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/features
    if osp.exists(feat_path + ".npy"):
        feats = np.load(feat_path + ".npy")
    else:
        generator, num = get_iterator(args)
        iterator = generator()

        feats = []
        for f in tqdm.tqdm(iterator, total=num):
            feats.append(f)

        del iterator
        del generator

        feats = np.concatenate(feats) # out feats.shape=[2196, 768]

        print(feats.shape)

        os.makedirs(args.save_dir, exist_ok=True) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep
        # np.save(feat_path, feats)

        gc.collect()
        torch.cuda.empty_cache()

    reload = False
    for spec in faiss_specs: # [faiss_spec(pca=0, norm=False, n_clus=128, sphere=False, spec_str='CLUS128')]
        print("Processing spec", spec)

        if reload: # false
            print("Reloading...")
            del feats
            gc.collect()
            feats = np.load(feat_path + ".npy")

        save_path = osp.join(args.save_dir, spec.spec_str) # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128'
        os.makedirs(save_path, exist_ok=True)
        d = feats.shape[-1] # d=768
        x = feats # [2196, 768]
        if spec.pca > 0: # = 0, not in
            print("Computing PCA")
            pca = faiss.PCAMatrix(d, spec.pca)
            pca.train(x)
            d = spec.pca
            b = faiss.vector_to_array(pca.b)
            A = faiss.vector_to_array(pca.A).reshape(pca.d_out, pca.d_in)
            np.save(osp.join(save_path, "pca_A"), A.T)
            np.save(osp.join(save_path, "pca_b"), b)
            print("Applying PCA")
            x = pca.apply_py(x)

        if spec.norm: # False, not in
            reload = spec.pca <= 0
            print("Normalizing")
            faiss.normalize_L2(x)

        #import ipdb; ipdb.set_trace()
        print("Computing kmeans")
        kmeans = faiss.Kmeans(
            d, # 768
            spec.n_clus, # 128
            niter=50,
            verbose=True,
            spherical=spec.sphere, # False
            max_points_per_centroid=feats.shape[0], # [2196, 768]
            gpu=True,
            nredo=3, # num of re-do, run the clustering this number of times, and keep the best centroids (selected according to clustering objective)
        ) # NOTE important! 
        kmeans.train(x) # <faiss.Kmeans object at 0x7f8e2c65e790>, x.shape=[2196, 768]
        np.save(osp.join(save_path, "centroids"), kmeans.centroids) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/centroids.npy
        del kmeans
        del x
        gc.collect()


if __name__ == "__main__":
    main()

    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv 
    # 这里是读取wav文件，然后基于wav2vec2的已有的checkpoint，来encode 这个wav，得到的是[frame.len, 768]，等读取了所有的wav文件，之后，把所有的features合并到一起，最后得到的是整体的[2196, 768] -> 然后，就可以继续k-means聚类了。
    # 这是基于k-means，来构造128个中心点：
    # outputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128/centroids.npy
