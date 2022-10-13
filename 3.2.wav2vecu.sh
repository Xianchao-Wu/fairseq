#########################################################################
# File Name: 3.2.wav2vecu.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Sat Oct  8 22:51:22 2022
#########################################################################
#!/bin/bash

script_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/scripts"

tsv_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads" # after vads

out_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep"
out_dir2="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2"

ckpt="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt"

isdone1=1
if [[ $isdone1 -eq 0 ]]
then
    bash $script_dir/prepare_audio.sh \
        $tsv_dir \
        $out_dir \
        $ckpt 512 11
    # TODO only has 12 layers, so "14" is not good...
fi

bash $script_dir/prepare_audio_v2.sh \
    $tsv_dir \
    $out_dir2 \
    $ckpt 64 11 

