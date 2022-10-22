#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

source_dir=$1
tgt_dir=$2
model=$3 # 预训练好的wav2vec2 model checkpoint

if [ -z "$4" ]
  then
    dim=512
  else
    dim=$4
fi

echo "using $dim dim for PCA"

if [ -z "$5" ]
  then
    layer=14 # TODO in the paper, the 15-th (start from 1) layer yielded the best results
  else
    layer=$5
fi

echo "extracting from layer $layer"

train_split="train"
valid_split="valid"
test_split="test"

all_splits=($train_split)

echo "source_dir=", $source_dir
echo "$source_dir/valid.tsv"
echo "$source_dir/test.tsv"

if [[ -f "$source_dir/valid.tsv" ]]; then
    #all_splits+=('valid')
    #all_splits+=($valid_split)
    all_splits="$all_splits $valid_split"
    echo "after valid:" $all_splits
fi

if [[ -f "$source_dir/test.tsv" ]]; then
    #all_splits+=('test')
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

# NOTE 1
isdone1=0
if [ $isdone1 -eq 0 ]
then
    echo $all_splits
    for split in $all_splits; do
        echo $split
    done

    for split in $all_splits; do
        echo $split
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_extract_features.py \
            $source_dir \
            --split $split \
            --save-dir $tgt_dir \
            --checkpoint $model \
            --layer $layer
    done
    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/{train,valid}.tsv
    # generates: train.lengths, train.npy (2196, 32) in 
    # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep
    # 这是通过调用wav2vec2，把一个个wav，表示成tensor的形式(re-scaled_frame_num, hidden_dimension)
    # train.tsv, valid.tsv 这两个文件是直接copy!
fi

isdone2=0
if [ $isdone2 -eq 0 ]
then
    # NOTE 2
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_cluster_faiss.py \
        $tgt_dir/${train_split}.tsv \
        --checkpoint $model \
        --save-dir $tgt_dir \
        -f "CLUS128" \
        --layer $layer \
        --sample-pct 1.0
    # inputs: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/train.tsv
    # 是对[2196, 768] 聚类
    # use k-means to cluster the encoded waves
    # output = centroids.npy in /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128
    # 这是使用k-means(K=128), 来对tensor进行聚类，得到centroids.npy，128个聚类中心点
fi

isdone3=0
if [ $isdone3 -eq 0 ]
then
    # NOTE 3
    for split in $all_splits; do
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/wav2vec_apply_cluster_faiss.py \
            $tgt_dir \
            --checkpoint $model \
            --path $tgt_dir/CLUS128 \
            --layer $layer \
            --split $split
        # inputs: 
        # output = train.src [after k-means, ids], train.tsv (file names) 
        # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/CLUS128
        # 为每个re-scaled frame，寻找和其距离最近的那个cluster-id，相当于把一个向量，映射到一个id
    done
fi

isdone4=0
if [ $isdone4 -eq 0 ]
then
    # NOTE 4
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/pca.py \
        $tgt_dir/${train_split}.npy \
        --output $tgt_dir/pca \
        --dim $dim #512 #16 #$dim
        # TODO 32 to 16 for debug only; and for real case is from 768 to 512
        # output = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/pca
        # files = 16_pca_A.npy and 16_pca_b.npy
        # 这是对wav2vec2的encoding output，进行pca抽取，例如，从32维度中，抽取出来重要的16个维度。
        # x' = x*A + b，这里是得到了A和b
fi

for split in $all_splits; do
    # NOTE 5
    ###dim=16 # for debug only TODO
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/apply_pca.py \
        $tgt_dir \
        --split $split \
        --save-dir $tgt_dir/precompute_pca$dim \
        --pca-path $tgt_dir/pca/${dim}_pca \
        --batch-size 1048000
    # output = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16
    # copied files: train.lengths and train.tsv; generated file: train.npy (share of [2196, 16])
    # done!
    # 这是根据上一步训练得到的A和b，应用到wav2vec2的表示张量上，得到的是[2196, 16]

    # NOTE 6
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/merge_clusters.py \
        $tgt_dir/precompute_pca$dim \
        --cluster-dir $tgt_dir/CLUS128 \
        --split $split \
        --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean \
        --pooling mean
    # output = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean
    # files = train.lengths (e.g., from 250 to 187 by average the continuous same ids, 1 1 1 -> 1 with duplicated count=3)
    # train.npy (with size from [2196, 16] to [1626, 16]) and train.tsv
    # 这是把k-means得到的id序列，相同的合并一下，例如1 1 1 -> 1，然后三个16维度的向量，求一下均值
    # 执行了这个步骤之后，train.npy的长度会减小，例如7个wav文件，整体长度从2196减小为1626.

    # NOTE 7
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/mean_pool.py \
        $tgt_dir/precompute_pca${dim}_cls128_mean \
        --save-dir $tgt_dir/precompute_pca${dim}_cls128_mean_pooled \
        --split $split
    # output = /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/train_vads/prep/precompute_pca16_cls128_mean_pooled
    # files = train.lengths (e.g., 61+94+73+145+31+229+182), train.npy (815, 16), train.tsv
    # 这个是做一下mean pooling，把相邻的两个frame合并一下，长度减半，从1626到815（有的长度会四舍五入）.
    #
done
