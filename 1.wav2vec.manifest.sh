#########################################################################
# File Name: 1.wav2vec.manifest.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Jun 10 00:20:19 2022
#########################################################################
#!/bin/bash

wavpath="/workspace/asr/csj/WAV"
destpath="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/csj"
ext="wav"
valid="0.01"

python -m ipdb examples/wav2vec/wav2vec_manifest.py $wavpath \
	--dest $destpath \
	--ext $ext \
	--valid-percent $valid
