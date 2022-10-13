# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import soundfile as sf
import torch
import torchaudio

from feature_utils import get_path_iterator, dump_feature
from fairseq.data.audio.audio_utils import get_features_or_waveform

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("dump_mfcc_feature")


class MfccFeatureReader(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate

    def read_audio(self, path, ref_len=None):
        wav = get_features_or_waveform(path, need_waveform=True, use_sample_rate=self.sample_rate)
        if ref_len is not None and abs(ref_len - len(wav)) > 160:
            logging.warning(f"ref {ref_len} != read {len(wav)} ({path})")
        return wav

    def get_feats(self, path, ref_len=None):
        x = self.read_audio(path, ref_len=ref_len)
        with torch.no_grad():
            x = torch.from_numpy(x).float()
            x = x.view(1, -1)

            mfccs = torchaudio.compliance.kaldi.mfcc(
                waveform=x,
                sample_frequency=self.sample_rate,
                use_energy=False,
            )  # (time, freq), e.g., (427, 13)
            mfccs = mfccs.transpose(0, 1)  # (freq, time), e.g., (13, 427)
            deltas = torchaudio.functional.compute_deltas(mfccs) # (13, 427)
            ddeltas = torchaudio.functional.compute_deltas(deltas) # (13, 427)
            concat = torch.cat([mfccs, deltas, ddeltas], dim=0) # (39, 427)
            concat = concat.transpose(0, 1).contiguous()  # (freq, time) -> (427, 39)
            return concat


def main(tsv_dir, split, nshard, rank, feat_dir, sample_rate):
    reader = MfccFeatureReader(sample_rate) # sample_rate=16000, 
    generator, num = get_path_iterator(f"{tsv_dir}/{split}.tsv", nshard, rank) # nshard=1, rank=0
    dump_feature(reader, generator, num, split, nshard, rank, feat_dir)
    # NOTE read, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/train.tsv
    # NOTE write to: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc
    #                train_0_1.len, train_0_1.npy 

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_dir")
    parser.add_argument("split")
    parser.add_argument("nshard", type=int)
    parser.add_argument("rank", type=int)
    parser.add_argument("feat_dir")
    parser.add_argument("--sample_rate", type=int, default=16000)
    args = parser.parse_args()
    logger.info(args)

    main(**vars(args))

