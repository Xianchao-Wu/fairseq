#########################################################################
# File Name: 6.gan.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Tue Oct 18 08:42:03 2022
#########################################################################
#!/bin/bash

PREFIX=w2v_unsup_gan_xp

# For wav2vec-U, audio features are pre-segmented
CONFIG_NAME=w2vu
# the real config file = /workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/config/gan/w2vu.yaml

timit_dir="/workspace/asr/timit_out_w2vu_512"

TASK_DATA="${timit_dir}/unmatched/feat/precompute_pca512_cls128_mean_pooled"
#/path/to/features/precompute_unfiltered_pca512_cls128_mean_pooled

# For wav2vec-U 2.0, use raw audio features
#CONFIG_NAME=w2vu2 # TODO
#TASK_DATA=/path/to/features/ # TODO

# Unpaired text input
TEXT_DATA=${timit_dir}/unmatched/phones  
# path to fairseq-preprocessed GAN data (phones dir)

KENLM_PATH=${timit_dir}/unmatched/phones/train_text_phn.04.bin  
# KenLM 4-gram phoneme language model (LM data = GAN data here)

config_gan="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/config/gan"

#PYTHONPATH=$FAIRSEQ_ROOT PREFIX=$PREFIX fairseq-hydra-train \
python -m ipdb fairseq_cli/hydra_train.py \
    -m --config-dir $config_gan \
    --config-name $CONFIG_NAME \
    task.data=${TASK_DATA} \
    task.text_data=${TEXT_DATA} \
    task.kenlm_path=${KENLM_PATH} \
    common.user_dir=${FAIRSEQ_ROOT}/examples/wav2vec/unsupervised \
    model.code_penalty=2 \
    model.gradient_penalty=1.5 \
    model.smoothness_weight=0.5 
    #'common.seed=range(0,5)'
    
#model.code_penalty=2,4 \
#model.gradient_penalty=1.5,2.0 \
#model.smoothness_weight=0.5,0.75,1.0 
#'common.seed=range(0,5)'
