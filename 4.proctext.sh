#########################################################################
# File Name: 4.proctext.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Thu Oct 13 22:01:49 2022
#########################################################################
#!/bin/bash

language="en"
txt_file="/workspace/asr/LibriSpeech/LibriSpeech/test-clean/test-clean-simp-lw.txt"
out_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt"
lid_model="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/lid.176.bin"
sil_prob=0.5

scripts="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/scripts"


#espeak, espeak-ng, G2P
# 1000 : min phoneme freq for filtering, too big for debugging...
bash $scripts/prepare_text.sh \
    $language \
    $txt_file \
    $out_dir \
    10 \
    "G2P" \
    $lid_model \
    $sil_prob
