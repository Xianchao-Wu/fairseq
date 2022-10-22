#########################################################################
# File Name: 5.timit.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Oct 17 16:35:43 2022
#########################################################################
#!/bin/bash

scripts="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/scripts"

timit_data="/workspace/asr/timit/data"
#output_dir="/workspace/asr/timit_out_w2vu" # final_dim=16
output_dir="/workspace/asr/timit_out_w2vu_512" # final_dim=512
w2v_model="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/wav2vec_small_100h.pt"

bash $scripts/prepare_timit.sh \
    $timit_data \
    $output_dir \
    $w2v_model
