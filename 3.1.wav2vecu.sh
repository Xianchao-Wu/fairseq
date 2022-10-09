#########################################################################
# File Name: 3.1.wav2vecu.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: 2022年10月08日 20時12分21秒
#########################################################################
#!/bin/bash

FAIRSEQ_ROOT="." 
RVAD_ROOT="/workspace/asr/wav2vec/rVADfast"
SCRIPT_ROOT="./examples/wav2vec/unsupervised/scripts/"

wav_dir="/workspace/asr/LibriSpeech/LibriSpeech/test-clean/test-clean-wav-debug"
out_dir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech"


# create a manifest file for the set original of audio files
# train.tsv is created
# "train" is a path!
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
	$wav_dir \
	--ext wav \
	--dest $out_dir/train \
	--valid-percent 0.0

# TODO change valid-percent
# detect human voices
cat $out_dir/train/train.tsv | \
	python $SCRIPT_ROOT/vads.py \
		-r $RVAD_ROOT > $out_dir/train/train.vads

# remove silence
python $SCRIPT_ROOT/remove_silence.py \
	--tsv $out_dir/train/train.tsv \
	--vads $out_dir/train/train.vads \
	--out $out_dir/train/vads

# build new train_vads.tsv basing on the new filtered wav files 
# "train_vads" is a path!
python $FAIRSEQ_ROOT/examples/wav2vec/wav2vec_manifest.py \
	$out_dir/train/vads \
	--ext wav \
	--dest $out_dir/train/train_vads \
	--valid-percent 0.2
