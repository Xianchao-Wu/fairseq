#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import numpy as np
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="converts words to phones adding optional silences around in between words"
    )
    parser.add_argument(
        "--sil-prob",
        "-s",
        type=float,
        default=0,
        help="probability of inserting silence between each word",
    )
    parser.add_argument(
        "--surround",
        action="store_true",
        help="if set, surrounds each example with silence",
    )
    parser.add_argument(
        "--lexicon",
        help="lexicon to convert to phones",
        required=True,
    )
    parser.add_argument("--in-file", type=str, help="input file to be processed")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    sil_prob = args.sil_prob
    surround = args.surround # True
    sil = "<SIL>"

    wrd_to_phn = {}

    with open(args.lexicon, "r") as lf: # NOTE read '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/lexicon_filtered.lst'
        for line in lf: # e.g., 'the\tDH AH\n'
            items = line.rstrip().split() # ['the', 'DH', 'AH']
            assert len(items) > 1, line
            assert items[0] not in wrd_to_phn, items
            wrd_to_phn[items[0]] = items[1:]
    
    print('adding SIL to {}'.format(args.in_file), file=sys.stderr)
    with open(args.in_file, 'r') as br: # NOTE read /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/lm.upper.lid.txt (plain sentences in each line -> to add sil to it) 
        for line in br.readlines():
            words = line.strip().split()

            if not all(w in wrd_to_phn for w in words):
                continue

            phones = []
            if surround:
                phones.append(sil)

            sample_sil_probs = None
            if sil_prob > 0 and len(words) > 1:
                sample_sil_probs = np.random.random(len(words) - 1) # 随机出来当前句子长度-1个概率值，用来指导是否增加sil到当前的那个word

            for i, w in enumerate(words):
                phones.extend(wrd_to_phn[w])
                if (
                    sample_sil_probs is not None
                    and i < len(sample_sil_probs)
                    and sample_sil_probs[i] < sil_prob
                ):
                    phones.append(sil)

            if surround:
                phones.append(sil)
            print(" ".join(phones)) # e.g., '<SIL> HH IY <SIL> T R AY D T UW <SIL> TH IH NG K <SIL> HH AW <SIL> IH T <SIL> K UH D <SIL> B IY <SIL>'

    print('done adding SIL', file=sys.stderr)

if __name__ == "__main__":
    main()
