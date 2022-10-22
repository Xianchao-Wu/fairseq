#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

timit_root=$1  # assume it is the upper-cased version
tgt_dir=$2
model=$3

set -eu

setups="matched unmatched"
splits="test valid train train_text"

tgt_dir=$(realpath $tgt_dir)
sph2wav=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
wav_dir=$tgt_dir/wav

mkdir -p $tgt_dir $wav_dir

###

isdone1=0
if [[ $isdone1 -eq 0 ]]; then
    find $timit_root/{TRAIN,TEST} -iname "*.WAV" > $tgt_dir/all_sph.flist # iname = case not sensitive! NOTE TODO
    cat $tgt_dir/all_sph.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).WAV#\1_\2#g' > $tgt_dir/all.uid # e.g., "FJDM2_SX232.wav" as one line of all.uid

    paste -d' ' $tgt_dir/{all_sph.flist,all.uid} | \
      awk -v sph2wav=$sph2wav -v wav_dir=$wav_dir '{print sph2wav " -f wav " $1 " > " wav_dir "/" $2 ".wav"}' \
      > $tgt_dir/sph2wav.sh
    bash $tgt_dir/sph2wav.sh
fi
# 上面是把6300个.WAV (sph格式)的文件，转成.wav。因为原本下载的数据，已经包括了.wav，所以会出：
# Input file /workspace/asr/timit/data/TEST/DR5/FJSA0/SX389.WAV.wav is not a valid SPHERE file
# 这样的问题，不重要了。现在/workspace/asr/timit_out_w2vu/wav 里面已经包括了所有需要的文件，6300个.wav。

#exit 0

###

isdone2=0
if [[ $isdone2 -eq 0 ]]; then
    cat $tgt_dir/all.uid | awk -v wav_dir=$wav_dir '{print $1" "wav_dir"/"$1".wav"}' | sort > $tgt_dir/all_wav.scp
    cut -d' ' -f2 $tgt_dir/all_wav.scp | xargs -I{} soxi -s {} > $tgt_dir/all.dur
    paste -d' ' $tgt_dir/{all_wav.scp,all.dur} > $tgt_dir/all_wav_dur.scp
fi

###

#rm $tgt_dir/{all.uid,all_sph.flist,sph2wav.sh}
isdone3=0
if [[ $isdone3 -eq 0 ]]; then
    find $timit_root/{TRAIN,TEST} -iname "*.PHN" > $tgt_dir/all_phn60.flist

    while read line; do
      if [ ! -f $line ]; then 
        >&2 echo "Cannot find transcription file '$line'" && exit 1;
      fi
      cut -f3 -d' ' "$line" | tr '\n' ' ' | perl -ape 's: *$:\n:;'
    done < $tgt_dir/all_phn60.flist > $tgt_dir/all.phn60
    # 这是把一个.PHN文件中的多行，合并成一行，得到如：h# sh iy hv ae dcl d y axr dcl d aa r kcl k s ux tcl en gcl g r iy s iy w aa sh epi w ao dx axr q ao l y iy axr h#
    # 上面的结果。
fi
#exit 0

###

isdone4=0
if [[ $isdone4 -eq 0 ]]; then
    # NOTE TODO 下面的 第一行是：从"/workspace/asr/timit/data/TRAIN/DR6/FJDM2/SA1.PHN"中提取出来"FJDM2_SA1"，是后面两个，用_连接！
    cat $tgt_dir/all_phn60.flist | sed -e 's#//*#/#g' -e 's#.*/\([^/]*\)/\([^/]*\).PHN#\1_\2#g' | \
      paste -d' ' - $tgt_dir/all.phn60 | \
      $KALDI_ROOT/egs/timit/s5/local/timit_norm_trans.pl -i - -m $KALDI_ROOT/egs/timit/s5/conf/phones.60-48-39.map -to 39 | \
      sort > $tgt_dir/all.phn
    echo "done preparing wav and 39-phone transcripts"
    # 上面是从60个phonemes，映射到39个。得到的是：FADG0_SA1 sil sh iy hh ae sil d y er sil d aa r sil k s uw sil t ih n sil g r iy s iy w aa sh sil w aa dx er aa l y ih er sil
fi
#exit 0

isdone5=0
if [[ $isdone5 -eq 0 ]]; then
    ###
    config_path="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/config"
    # setups="matched unmatched"
    # matched: 这里面的train.uid和train_text.uid是一样的;
    # unmatched: train.uid和train_text.uid是不同的！ TODO
    for s in $setups; do
      mkdir -p $tgt_dir/$s
      # 
      # splits="test valid train train_text"
      for x in $splits; do
        uid_path=${config_path}/timit_${s}/${x}.uid
        grep -w -f $uid_path $tgt_dir/all.phn | cut -d' ' -f2- > $tgt_dir/$s/$x.phn # 只保留phoneme sequence: sil ah s uw m f aa r ih sil z ae m sil p uh l ah s ih sil ch uw ey sh n w eh er f aa r m hh eh z ah sil p ae sil k iy ng sh eh sil d sil ae n sil d f iy l sil s sil
        ln -sf $(realpath $tgt_dir/$s/$x.phn) $tgt_dir/$s/$x.wrd # 这只是增加一个快捷方式，train_text.phn -> train_text.wrd (.wrd是.phn的alias)
        
        echo "/" > $tgt_dir/$s/$x.tsv &&  grep -w -f $uid_path $tgt_dir/all_wav_dur.scp | cut -d' ' -f2- | sed 's# #\t#'  >> $tgt_dir/$s/$x.tsv # 这是构建 train_text.tsv，第一行是一个/，然后是如：/workspace/asr/timit_out_w2vu/wav/FAEM0_SI1392.wav      76186
      done
      
      for x in $splits; do
        cat $tgt_dir/$s/$x.phn
      done | tr ' ' '\n' | sort -u | awk '{print $1" "1}' > $tgt_dir/$s/dict.phn.txt # 包括了39行，aa 1; ae 1; ..., z 1
      ln -sf $(realpath $tgt_dir/$s/dict.phn.txt) $tgt_dir/$s/dict.wrd.txt # 这是建立了一个对dict.phn.txt的别名 dict.wrd.txt
    done
    echo "done preparing unmatched and matched setups for TIMIT"
fi
#exit 0

scripts="/workspace/asr/wav2vec/fairseq/examples/wav2vec/unsupervised/scripts"
# setups="matched unmatched"
setups="unmatched"
for s in $setups; do

  bash $scripts/prepare_audio.sh $tgt_dir/$s $tgt_dir/$s/feat $model 512 11
  
  #exit 0

  lm_dir=$tgt_dir/$s/phones
  fst_dir=$tgt_dir/$s/fst/phn_to_phn

  python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py --dataset-impl mmap --trainpref $tgt_dir/$s/train_text.phn --workers 10 --only-source --destdir $lm_dir --srcdict $tgt_dir/$s/dict.phn.txt

  #exit 0

  $KENLM_ROOT/lmplz -o 3 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.03.arpa
  $KENLM_ROOT/build_binary $lm_dir/train_text_phn.03.arpa $lm_dir/train_text_phn.03.bin
  $KENLM_ROOT/lmplz -o 4 < $tgt_dir/$s/train_text.phn --discount_fallback >$lm_dir/train_text_phn.04.arpa
  $KENLM_ROOT/build_binary $lm_dir/train_text_phn.04.arpa $lm_dir/train_text_phn.04.bin
  
  python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$fst_dir lm_arpa=$lm_dir/train_text_phn.03.arpa data_dir=$tgt_dir/$s in_labels=phn
  #exit 0

done

echo "done preprocessing audio and text for wav2vec-U"
