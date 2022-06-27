#########################################################################
# File Name: 2.train.wav2vec2.base.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Fri Jun 10 01:00:51 2022
#########################################################################
#!/bin/bash

. ./path.sh || exit 1

data="/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/csj"
confdir="/workspace/asr/wav2vec/fairseq/examples/wav2vec/config/pretraining"
confname="wav2vec2_base_csj"

#python -m ipdb fairseq_cli/hydra_train.py \
python fairseq_cli/hydra_train.py \
	task.data=$data \
	--config-dir $confdir \
	--config-name $confname 
