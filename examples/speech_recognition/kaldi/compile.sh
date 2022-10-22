#########################################################################
# File Name: compile.sh
# Author: Xianchao Wu
# mail: xianchaow@nvidia.com
# Created Time: Mon Oct 17 15:28:35 2022
#########################################################################
#!/bin/bash

#c++ -I/workspace/asr/wav2vec/kaldi/src \
c++ -I/workspace/asr/wav2vec/kaldi/src \
    -I/workspace/asr/wav2vec/kaldi/tools/openfst-1.7.2/include \
    -L/workspace/asr/wav2vec/kaldi/src/lib \
    add-self-loop-simple.cc \
    -lkaldi-base \
    -lkaldi-fstext \
    -o add-self-loop-simple
