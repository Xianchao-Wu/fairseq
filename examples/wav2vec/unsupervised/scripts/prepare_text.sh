#!/usr/bin/env zsh
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

lg=$1 # langauge flag, e.g., "en"
text_path=$2
target_dir=$3
min_phones=$4
phonemizer=$5
lid_path=$6
sil_prob=$7

if [ -z "$lid_path" ]; then
  lid_path="lid.187.bin"
fi

ph_lg=${lg:l}
if test "$lg" = 'fr'; then
  ph_lg='fr-fr'
elif test "$lg" = 'en'; then
  ph_lg='en-us'
elif test "$lg" = 'pt'; then
  ph_lg='pt-br'
fi

ESPEAK_PATH=''
if test "$phonemizer" = 'espeak'; then
  ESPEAK_PATH=$(which espeak)
elif test "$phonemizer" = 'espeak-ng'; then
  ESPEAK_PATH=$(which espeak-ng)
elif test "$phonemizer" = 'G2P'; then
  ESPEAK_PATH=''
else
  echo "Unknown phonemizer $phonemizer. Valid options are espeak, espean-ng and G2P"
  exit 1
fi

echo $lg
echo $ph_lg
echo $text_path
echo $target_dir
echo "min phone seen threshold is $min_phones"

mkdir -p $target_dir

isdone1=0
if [[ $isdone1 -eq 0 ]]; then
    # NOTE 1 逐句语言识别
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/normalize_and_filter_text.py \
        --lang $lg \
        --fasttext-model $lid_path \
        --in-file $text_path > $target_dir/before.lm.upper.lid.txt
        #< $text_path #
    cat $target_dir/before.lm.upper.lid.txt | grep -v '\-\-\-' > $target_dir/lm.upper.lid.txt
    # >! is for overwrite even exists...
    # 识别语言种类（按照句子）
    # input=/workspace/asr/LibriSpeech/LibriSpeech/test-clean/test-clean-simp-lw.txt
    # outputs=/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/
    #         before.lm.upper.lid.txt, lm.upper.lid.txt

    # NOTE 2
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py \
        --dataset-impl mmap \
        --trainpref $target_dir/lm.upper.lid.txt \
        --only-source \
        --destdir $target_dir \
        --thresholdsrc 2 \
        --padding-factor 1 \
        --dict-only
    # case 1 
        # input=/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/lm.upper.lid.txt
        # output=/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/dict.txt 
        # 构建词典

    cut -f1 -d' ' $target_dir/dict.txt | grep -v -x '[[:punct:]]*' | grep -Pv '\d\d\d\d\d+' > $target_dir/words.txt
    # from /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt
    # dict.txt to words.txt

    # NOTE 3
    echo $ESPEAK_PATH

    if [ -z "$ESPEAK_PATH" ]; then
        # this is using G2P phoneme processor:
        python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/g2p_wrd_to_phn.py \
            --compact \
            --in-file $target_dir/words.txt > $target_dir/phones.txt
        # here NOTE
        # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt
        # 'words.txt' -> 'phones.txt', 
        # 这是为每个英文单词，构造出来phoneme sequence，音素序列
    else
        # echoing 1 into corpus will prevent the mismatch lines between 
        # lexicon and phones in case the phonemizer fails
        one=$(echo "1" | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -p ' ' -w '' -l $ph_lg --language-switch remove-flags)

        sed 's/$/ 1/' $target_dir/words.txt | PHONEMIZER_ESPEAK_PATH=$ESPEAK_PATH phonemize -o $target_dir/phones.txt -p ' ' -w '' -l $ph_lg -j 70 --language-switch remove-flags

        echo "one is ${one}"

        sed -i "s/${one}$//" $target_dir/phones.txt
    fi

    paste $target_dir/words.txt $target_dir/phones.txt > $target_dir/lexicon.lst # 这是把两个文件拼接成一个，变成两列内容了。

    # NOTE 4
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py \
        --dataset-impl mmap \
        --trainpref $target_dir/phones.txt \
        --only-source \
        --destdir $target_dir/phones \
        --thresholdsrc $min_phones \
        --padding-factor 1 \
        --dict-only
    # 上面的是，根据phones.txt构造出来一个新的dict: phones/dict.txt
    # case 2
    # NOTE 5
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/filter_lexicon.py \
        -d $target_dir/phones/dict.txt \
        --in-file $target_dir/lexicon.lst > $target_dir/lexicon_filtered.lst
    # 上面的是，根据phones/dict.txt，来过滤lexicon.lst，得到的是lexicon_filtered.lst

    # NOTE 6
    python $FAIRSEQ_ROOT/examples/wav2vec/unsupervised/scripts/phonemize_with_sil.py \
        -s $sil_prob \
        --surround \
        --lexicon $target_dir/lexicon_filtered.lst \
        --in-file $target_dir/lm.upper.lid.txt > $target_dir/phones/lm.phones.filtered.txt

        # input: lexicon_filtered.lst, lm.upper.lid.txt 
        # output: phones/lm.phones.filtered.txt
        # 根据过滤后的词典，来为文本集合lm.upper.lid.txt中的每个句子的词和词之间，增加 <SIL>

    cp $target_dir/phones/dict.txt $target_dir/phones/dict.phn.txt
    echo "<SIL> 0" >> $target_dir/phones/dict.phn.txt

    # NOTE 7
    python $FAIRSEQ_ROOT/fairseq_cli/preprocess.py \
        --dataset-impl mmap \
        --trainpref $target_dir/phones/lm.phones.filtered.txt \
        --only-source \
        --destdir $target_dir/phones \
        --srcdict $target_dir/phones/dict.phn.txt
        #--workers 70 \
    # case 3
    # input: phones/ lm.phones.filtered.txt, dict.phn.txt
    # output: phones/train.bin and phones/train.idx

    $KENLM_ROOT/lmplz -o 4 < $target_dir/lm.upper.lid.txt --discount_fallback --prune 0 0 0 3 > $target_dir/kenlm.wrd.o40003.arpa
    $KENLM_ROOT/build_binary $target_dir/kenlm.wrd.o40003.arpa $target_dir/kenlm.wrd.o40003.bin

    # input: lm.upper.lid.txt
    # output: kenlm.wrd.o40003.arpa, kenlm.wrd.o40003.bin
fi

isdone2=0
if [[ $isdone2 -eq 0 ]]; then
    # NOTE 8 lg="en"
    # TODO to check more details... 
    lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py \
        kaldi_root=$KALDI_ROOT \
        fst_dir=$target_dir/fst/phn_to_words_sil \
        lm_arpa=$target_dir/kenlm.wrd.o40003.arpa \
        wav2letter_lexicon=$target_dir/lexicon_filtered.lst \
        data_dir=$target_dir/phones \
        in_labels=phn \
        "blank_symbol='<SIL>'"

    # NOTE 9
    lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py \
        kaldi_root=$KALDI_ROOT \
        fst_dir=$target_dir/fst/phn_to_words \
        lm_arpa=$target_dir/kenlm.wrd.o40003.arpa \
        wav2letter_lexicon=$target_dir/lexicon_filtered.lst \
        data_dir=$target_dir/phones \
        in_labels=phn
fi

isdone3=0
if [[ $isdone3 -eq 0 ]]; then
    $KENLM_ROOT/lmplz -o 4 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.04.arpa

    $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.04.arpa $target_dir/phones/lm.phones.filtered.04.bin
    $KENLM_ROOT/lmplz -o 6 < $target_dir/phones/lm.phones.filtered.txt --discount_fallback > $target_dir/phones/lm.phones.filtered.06.arpa
    $KENLM_ROOT/build_binary $target_dir/phones/lm.phones.filtered.06.arpa $target_dir/phones/lm.phones.filtered.06.bin
fi

lg=$lg python $FAIRSEQ_ROOT/examples/speech_recognition/kaldi/kaldi_initializer.py kaldi_root=$KALDI_ROOT fst_dir=$target_dir/fst/phn_to_phn_sil lm_arpa=$target_dir/phones/lm.phones.filtered.06.arpa data_dir=$target_dir/phones in_labels=phn "blank_symbol='<SIL>'"

