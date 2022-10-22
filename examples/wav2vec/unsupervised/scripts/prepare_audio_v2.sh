#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

source_dir=$1
tgt_dir=$2
model=$3

if [ -z "$4" ]
  then
    dim=64
  else
    dim=$4
fi

echo "using $dim clusters for auxilary target"

if [ -z "$5" ]
  then
    layer=11 #14 TODO
  else
    layer=$5
fi

echo "extracting from layer $layer"

train_split="train"
valid_split="valid"
test_split="test"

all_splits=($train_split)

if [[ -f "$source_dir/valid.tsv" ]]; then
    all_splits="$all_splits $valid_split"
    echo "after valid:" $all_splits
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    all_splits="$all_splits $test_split"
    echo "after test:" $all_splits
fi

echo "processing splits: $all_splits"

mkdir -p $tgt_dir

cp $source_dir/*.tsv $tgt_dir
cp $source_dir/*.wrd $tgt_dir
cp $source_dir/*.ltr $tgt_dir
cp $source_dir/*.phn $tgt_dir
cp $source_dir/dict* $tgt_dir

#setopt shwordsplit
#

isdone1=0
if [[ $isdone1 -eq 0 ]]
then
    for split in $all_splits; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py \
            $source_dir \
            --split $split \
            --save-dir $tgt_dir \
            --checkpoint $model \
            --layer $layer
    done
    # 这个和prepare_audio.sh里面的第一步，完全一样。
    
    mkdir -p $tgt_dir/mfcc

    # Consider spliting corpus into chuncks for large corpus, see HuBERT preprocessing for more details
    python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_mfcc_feature.py \
        $tgt_dir $train_split 1 0 $tgt_dir/mfcc
    # 这是使用kaldi的mfcc的方法，构造train.tsv中的，每个当前wav的mfcc特征。
    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2
    #         train.tsv
    #
    # outputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc
    # train_0_1.len, train_0_1.npy
fi

isdone2=0
if [[ $isdone2 -eq 0 ]]
then
    feat_dir=$tgt_dir/mfcc
    km_path=$tgt_dir/mfcc/cls$dim

    python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/learn_kmeans.py \
        $feat_dir \
        $train_split \
        1 \
        $km_path \
        $dim \
        --seed 42 \
        --percent -1 \
        --init "k-means++" \
        --max_iter 100 \
        --batch_size 10000 \
        --tol 0.0 \
        --max_no_improvement 100 \
        --n_init 20 \
        --reassignment_ratio 0.0 

    python $FAIRSEQ_ROOT/examples/hubert/simple_kmeans/dump_km_label.py \
        $tgt_dir/mfcc $train_split $tgt_dir/mfcc/cls$dim 1 0 $tgt_dir/mfcc/cls${dim}_idx
fi

cp $tgt_dir/mfcc/cls${dim}_idx/${train_split}_0_1.km $tgt_dir/$train_split.km
# /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/mfcc/cls64_idx/train_0_1.km
# ->
# /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prepv2/train.km
