#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
from shutil import copyfile

from npy_append_array import NpyAppendArray
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
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    parser.add_argument('--checkpoint', type=str, help='checkpoint for wav2vec ctc model', required=True)
    parser.add_argument('--layer', type=int, default=14, help='which layer to use')
    # fmt: on

    return parser


class Wav2VecFeatureReader(object):
    def __init__(self, cp_file, layer):
        #import ipdb; ipdb.set_trace()
        # TODO
        model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [cp_file] # checkpoint file: '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt' 
        )
        model = model[0]
        model.eval()
        model.cuda()
        self.model = model # <class 'fairseq.models.wav2vec.wav2vec2_asr.Wav2VecCtc'>
        self.task = task # <fairseq.tasks.audio_pretraining.AudioPretrainingTask object at 0x7efba08e7160>
        self.layer = layer # 11

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc) 
        # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads/test-clean-wav-debug/908-31957-0005.wav, x.shape=(54240,) 读取语音数据，采样率16K

        with torch.no_grad():
            source = torch.from_numpy(x).float().cuda() # numpy to tensor, torch.Size([54240])
            if self.task.cfg.normalize: # False
                assert source.dim() == 1, source.dim()
                with torch.no_grad():
                    source = F.layer_norm(source, source.shape)
            source = source.view(1, -1) # torch.Size([1, 54240])
            #import ipdb; ipdb.set_trace()
            #m_res = self.model(source=source, mask=False, features_only=True, layer=self.layer) 
            # NOTE TODO changed from mask=False to padding_mask=None!
            # appending is better...
            m_res = self.model(source=source, padding_mask=None, features_only=True, layer=self.layer) 
            # dict_keys(['encoder_out', 'padding_mask', 'layer_results']). 

            #return m_res["x"].squeeze(0).cpu()
            # TODO
            return m_res['encoder_out'].squeeze(1).cpu() # [169, 1, 32] to [169, 32]
            # TODO, m_res['layer_results'][11=out.layer][0].shape = [169, 1, 768]

def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp: 
        # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/train.tsv'

        lines = fp.read().split("\n")
        root = lines.pop(0).strip() 
        # root path, e.g., '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads'

        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0] 
        # 完整的wav文件的路径的集合，
        # e.g., ['/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/vads/test-clean-wav-debug/908-31957-0005.wav', ...]

        num = len(files)
        reader = Wav2VecFeatureReader(args.checkpoint, args.layer)
        # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt, layer=11
        def iterate():
            for fname in files:
                w2v_feats = reader.get_feats(fname)
                yield w2v_feats # shape=[169, 32=out.dim after a 768-to-32 projection], TODO

    return iterate, num # 返回的是，iterate=生成器函数, num=7=文件数量


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True) 
    # save_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep"

    # dest = 目标路径，
    # e.g., '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train' 
    def create_files(dest): 
        # dest=/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train

        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv") 
        # args.data="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads". 
        # 这是把文件/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/train.tsv 
        # 复制为/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv

        if osp.exists(osp.join(args.data, args.split) + ".wrd"): # 目前没有
            copyfile(osp.join(args.data, args.split) + ".wrd", dest + ".wrd")
        if osp.exists(osp.join(args.data, args.split) + ".phn"): # 目前没有
            copyfile(osp.join(args.data, args.split) + ".phn", dest + ".phn")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy") 
        # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.npy'
        return npaa # <npy_append_array.npy_append_array.NpyAppendArray object at 0x7feb238dc9d0>

    save_path = osp.join(args.save_dir, args.split) 
    # save_path='/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train'
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f:
        for w2v_feats in tqdm.tqdm(iterator, total=num):
            print(len(w2v_feats), file=l_f)

            if len(w2v_feats) > 0:
                npaa.append(w2v_feats.numpy()) 
                # finally, [2196, 32] and save to train.npy in 
                # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep


if __name__ == "__main__":
    main()

    # input files = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/train.tsv

    # output files = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep
    #     train.lengths - 169, 250, 214, 391, 85, 599, 488
    #     train.npy - tensor with shape = [2196, 32]
    #     train.tsv - direct copy
    
    #     valid.lengths
    #     valid.npy
    #     valid.tsv - direct copy


    ###
    # this is for prepare_audio_v2.sh
    # input files: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads
    #              train.tsv, valid.tsv

    # output files: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2
    # output files: copied: train.tsv
    #               train.lengths, train.npy

    # output files: copied: valid.tsv
    #               valid.lengths, valid.npy

