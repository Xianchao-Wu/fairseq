#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

from g2p_en import G2p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compact",
        action="store_true",
        help="if set, compacts phones",
    )
    parser.add_argument(
        "--in-file",
        type=str,
        help="input file",
    )
    args = parser.parse_args()
    #import ipdb; ipdb.set_trace()

    compact = args.compact

    wrd_to_phn = {}
    g2p = G2p() # <g2p_en.g2p.G2p object at 0x7f7061c46ca0>

    print('word2phoneme (G2P): reading file={}'.format(args.in_file), file=sys.stderr)
    with open(args.in_file, 'r') as br: # '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/words.txt' NOTE read file
        #for line in sys.stdin:
        for line in br.readlines():
            words = line.strip().split()
            phones = []
            for w in words:
                if w not in wrd_to_phn:
                    wrd_to_phn[w] = g2p(w) # w='the', g2p(w)=['DH', 'AH0'], TODO
                    if compact:
                        wrd_to_phn[w] = [
                            p[:-1] if p[-1].isnumeric() else p for p in wrd_to_phn[w]
                        ]
                phones.extend(wrd_to_phn[w]) # from ['DH', 'AH0'] to ['DH', 'AH'], removed number!
            try:
                print(" ".join(phones)) # 'DH AH'
            except:
                print(wrd_to_phn, words, phones, file=sys.stderr)
                raise
    print('done word2phoneme', file=sys.stderr)


if __name__ == "__main__":
    main()

