#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

sys.path.append('/workspace/asr/wav2vec/fairseq')
from fairseq.data import Dictionary


def get_parser():
    parser = argparse.ArgumentParser(
        description="filters a lexicon given a unit dictionary"
    )
    parser.add_argument("-d", "--unit-dict", help="unit dictionary", required=True)
    parser.add_argument("--in-file", type=str, help="input file")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    d = Dictionary.load(args.unit_dict) # <fairseq.data.dictionary.Dictionary object at 0x7f96212dce20>; NOTE read file: '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/dict.txt'
    symbols = set(d.symbols) # e.g., symbols={'AA', 'EY', 'CH', 'W', 'IY', 'V', ... }

    print('filter {} by {}'.format(args.in_file, args.unit_dict), file=sys.stderr)
    with open(args.in_file, 'r') as br: # NOTE read file: '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/lexicon.lst' - to be filtered.
        for line in br.readlines():
        #for line in sys.stdin:
            items = line.rstrip().split()
            skip = len(items) < 2
            for x in items[1:]:
                if x not in symbols:
                    skip = True
                    break
            if not skip:
                print(line, end="")
    print('done filtering', file=sys.stderr)


if __name__ == "__main__":
    main()
