#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import fasttext as ft
import os
import regex
import sys


def get_parser():
    parser = argparse.ArgumentParser(
        description="reads text from stdin and outputs normalized, lid-filtered version to stdout"
    )
    parser.add_argument(
        "--fasttext-model",
        type=str,
        help="path to fasttext model",
        default="lid.187.bin",
    )
    parser.add_argument("--lang", type=str, help="language id", required=True)
    parser.add_argument("--in-file", type=str, help="input txt file", required=True)
    parser.add_argument(
        "--lid-threshold",
        type=float,
        help="threshold for this lang id probability",
        default=0.4,
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    filter_r = regex.compile(r"[^\p{L}\p{N}\p{M}\' \-]")

    lg = args.lang.lower() # 'en' - target language id/flag
    lg_label = f"__label__{lg}" # '__label__en'
    thresh = args.lid_threshold # 0.4

    if os.path.exists(args.fasttext_model):
        model = ft.load_model(args.fasttext_model) # <fasttext.FastText._FastText object at 0x7fcf742ad460>
    else:
        print(
            f"fasttext language id model {args.fasttext_model} not found. Proceeding without language filtering. "
            f"To enable language filtering, please download the latest language id model "
            f"from https://fasttext.cc/docs/en/language-identification.html",
            file=sys.stderr,
        )
        model = None

    with open(args.in_file, 'r') as br:
        #for line in sys.stdin:
        for line in br.readlines():
            line = line.strip()
            line = filter_r.sub(" ", line)
            line = " ".join(line.split())

            if model is not None:
                lid, prob = model.predict(line, k=100) # language identification; lid=id of 100 languages, such as ('__label__en', '__label__de', ...); prob=prob of 100 languages, such as array([9.32203829e-01, 7.40128104e-03, 4.11371514e-03, 3.52801336e-03,
                try:
                    target_idx = lid.index(lg_label) # e.g., 0 (for en)
                except ValueError:
                    continue
                if target_idx == 0 or prob[target_idx] >= thresh:
                    print(line) # 验证ok了，这的确是目标语言en的一个句子，哦耶~~
            else:
                print(line)


if __name__ == "__main__":
    main()
