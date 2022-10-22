#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
import typing as tp
from argparse import Namespace
from itertools import zip_longest

sys.path.append('/workspace/asr/wav2vec/fairseq')

from fairseq import options, tasks, utils
from fairseq.binarizer import (
    AlignmentDatasetBinarizer,
    FileBinarizer,
    VocabularyDatasetBinarizer,
)
from fairseq.data import Dictionary

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")

#####################################################################
# file name tools
#####################################################################


def _train_path(lang, trainpref):
    return "{}{}".format(trainpref, ("." + lang) if lang else "")


def _file_name(prefix, lang):
    fname = prefix
    if lang is not None:
        fname += ".{lang}".format(lang=lang)
    return fname


def _dest_path(prefix, lang, destdir):
    return os.path.join(destdir, _file_name(prefix, lang))


def _dict_path(lang, destdir):
    return _dest_path("dict", lang, destdir) + ".txt"


def dataset_dest_prefix(args, output_prefix, lang):
    base = os.path.join(args.destdir, output_prefix)
    if lang is not None:
        lang_part = f".{args.source_lang}-{args.target_lang}.{lang}"
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = f".{args.source_lang}-{args.target_lang}"

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    return "{}.{}".format(dataset_dest_prefix(args, output_prefix, lang), extension)


#####################################################################
# dictionary tools
#####################################################################


def _build_dictionary(
    filenames,
    task,
    args,
    src=False,
    tgt=False,
):
    assert src ^ tgt
    return task.build_dictionary(
        filenames,
        workers=args.workers,
        threshold=args.thresholdsrc if src else args.thresholdtgt,
        nwords=args.nwordssrc if src else args.nwordstgt,
        padding_factor=args.padding_factor,
    )


#####################################################################
# bin file creation logic
#####################################################################


def _make_binary_dataset(
    vocab: Dictionary, # case 3, <fairseq.data.dictionary.Dictionary object at 0x7f7e14f78880>
    input_prefix: str, # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt
    output_prefix: str, # 'train'
    lang: tp.Optional[str], # None
    num_workers: int, # 1
    args: Namespace, # 一大坨来自命令行的参数集合
):
    logger.info("[{}] Dictionary: {} types".format(lang, len(vocab))) # case 3, 2022-10-14 16:20:14 | INFO | fairseq_cli.preprocess | [None] Dictionary: 44 types

    binarizer = VocabularyDatasetBinarizer(
        vocab,
        append_eos=True,
    ) # <fairseq.binarizer.VocabularyDatasetBinarizer object at 0x7f7ceaae9fd0>

    input_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "") # NOTE read '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt'
    full_output_prefix = dataset_dest_prefix(args, output_prefix, lang) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/train

    final_summary = FileBinarizer.multiprocess_dataset(
        input_file, # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt
        args.dataset_impl, # 'mmap'
        binarizer, # <fairseq.binarizer.VocabularyDatasetBinarizer object at 0x7f7ceaae9fd0>
        full_output_prefix, # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/train
        vocab_size=len(vocab), # 44
        num_workers=num_workers, # 1
    )

    logger.info(f"[{lang}] {input_file}: {final_summary} (by {vocab.unk_word})")
    # 2022-10-14 16:22:38 | INFO | fairseq_cli.preprocess | [None] /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt: 879 sents, 43989 tokens, 0.0% replaced (by <unk>)

def _make_binary_alignment_dataset(
    input_prefix: str, output_prefix: str, num_workers: int, args: Namespace
):

    binarizer = AlignmentDatasetBinarizer(utils.parse_alignment)

    input_file = input_prefix
    full_output_prefix = dataset_dest_prefix(args, output_prefix, lang=None)

    final_summary = FileBinarizer.multiprocess_dataset(
        input_file,
        args.dataset_impl,
        binarizer,
        full_output_prefix,
        vocab_size=None,
        num_workers=num_workers,
    )

    logger.info(
        "[alignments] {}: parsed {} alignments".format(
            input_file, final_summary.num_seq
        )
    )


#####################################################################
# routing logic
#####################################################################


def _make_dataset(
    vocab: Dictionary, # <fairseq.data.dictionary.Dictionary object at 0x7f7e14f78880>
    input_prefix: str, # case 3, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt
    output_prefix: str, # case 3, 'train'
    lang: tp.Optional[str], # None
    args: Namespace, # 命令行
    num_workers: int, # 1
):
    if args.dataset_impl == "raw":
        # Copy original text file to destination folder
        output_text_file = _dest_path(
            output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
            lang,
            args.destdir,
        )
        shutil.copyfile(_file_name(input_prefix, lang), output_text_file)
    else:
        _make_binary_dataset(
            vocab, input_prefix, output_prefix, lang, num_workers, args
        )

# lang=None, vocab=<fairseq.data.dictionary.Dictionary object at 0x7f7e14f78880>; args=命令行参数
def _make_all(lang, vocab, args):
    if args.trainpref: # case 3, '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/lm.phones.filtered.txt'
        _make_dataset(
            vocab, args.trainpref, "train", lang, args=args, num_workers=args.workers
        )
    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            outprefix = "valid{}".format(k) if k > 0 else "valid"
            _make_dataset(
                vocab, validpref, outprefix, lang, args=args, num_workers=args.workers
            )
    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            outprefix = "test{}".format(k) if k > 0 else "test"
            _make_dataset(
                vocab, testpref, outprefix, lang, args=args, num_workers=args.workers
            )


def _make_all_alignments(args):
    if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
        _make_binary_alignment_dataset(
            args.trainpref + "." + args.align_suffix,
            "train.align",
            num_workers=args.workers,
            args=args,
        )
    if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
        _make_binary_alignment_dataset(
            args.validpref + "." + args.align_suffix,
            "valid.align",
            num_workers=args.workers,
            args=args,
        )
    if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
        _make_binary_alignment_dataset(
            args.testpref + "." + args.align_suffix,
            "test.align",
            num_workers=args.workers,
            args=args,
        )


#####################################################################
# align
#####################################################################


def _align_files(args, src_dict, tgt_dict):
    assert args.trainpref, "--trainpref must be set if --alignfile is specified"
    src_file_name = _train_path(args.source_lang, args.trainpref)
    tgt_file_name = _train_path(args.target_lang, args.trainpref)
    freq_map = {}
    with open(args.alignfile, "r", encoding="utf-8") as align_file:
        with open(src_file_name, "r", encoding="utf-8") as src_file:
            with open(tgt_file_name, "r", encoding="utf-8") as tgt_file:
                for a, s, t in zip_longest(align_file, src_file, tgt_file):
                    si = src_dict.encode_line(s, add_if_not_exist=False)
                    ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                    ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                    for sai, tai in ai:
                        srcidx = si[int(sai)]
                        tgtidx = ti[int(tai)]
                        if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                            assert srcidx != src_dict.pad()
                            assert srcidx != src_dict.eos()
                            assert tgtidx != tgt_dict.pad()
                            assert tgtidx != tgt_dict.eos()
                            if srcidx not in freq_map:
                                freq_map[srcidx] = {}
                            if tgtidx not in freq_map[srcidx]:
                                freq_map[srcidx][tgtidx] = 1
                            else:
                                freq_map[srcidx][tgtidx] += 1
    align_dict = {}
    for srcidx in freq_map.keys():
        align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)
    with open(
        os.path.join(
            args.destdir,
            "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
        ),
        "w",
        encoding="utf-8",
    ) as f:
        for k, v in align_dict.items():
            print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


#####################################################################
# MAIN
#####################################################################


def main(args):
    # setup some basic things
    utils.import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True) # case 1: 
    # case 2, 3: '/workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones'
    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(args.destdir, "preprocess.log"),
        )
    )
    logger.info(args)

    assert (
        args.dataset_impl != "huffman"
    ), "preprocessing.py doesn't support Huffman yet, use HuffmanCodeBuilder directly."

    # build dictionaries

    target = not args.only_source # False

    if not args.srcdict and os.path.exists(_dict_path(args.source_lang, args.destdir)): # 如果文件已经存在，就报异常
        raise FileExistsError(_dict_path(args.source_lang, args.destdir))

    if (
        target
        and not args.tgtdict
        and os.path.exists(_dict_path(args.target_lang, args.destdir))
    ):
        raise FileExistsError(_dict_path(args.target_lang, args.destdir))

    task = tasks.get_task(args.task) # NOTE, default is [case 1,2,3] <class 'fairseq.tasks.translation.TranslationTask'> ...

    if args.joined_dictionary: # not in
        assert (
            not args.srcdict or not args.tgtdict
        ), "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert (
                args.trainpref
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = _build_dictionary(
                {
                    _train_path(lang, args.trainpref)
                    for lang in [args.source_lang, args.target_lang]
                },
                task=task,
                args=args,
                src=True,
            )
        tgt_dict = src_dict
    else:
        if args.srcdict: # not in (case1,2); case3: in NOTE read: /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/dict.phn.txt
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert (
                args.trainpref # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/lm.upper.lid.txt, NOTE read file
            ), "--trainpref must be set if --srcdict is not specified"
            src_dict = _build_dictionary(
                [_train_path(args.source_lang, args.trainpref)], # NOTE read lm.upper.lid.txt file as input for building dict; NOTE case 2 read /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones.txt and build dict; 
                task=task,
                args=args,
                src=True,
            ) # <fairseq.data.dictionary.Dictionary object at 0x7f97741f19a0> TODO

        if target: # False
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert (
                    args.trainpref
                ), "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = _build_dictionary(
                    [_train_path(args.target_lang, args.trainpref)],
                    task=task,
                    args=args,
                    tgt=True,
                )
        else:
            tgt_dict = None

    # save dictionaries

    src_dict.save(_dict_path(args.source_lang, args.destdir)) # /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/dict.txt NOTE, write to dict file (also with preprocess.log); NOTE case 2, write to /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/dict.txt (also with preprocess.log); NOTE case 3, /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones/dict.txt [dict.txt = dict.phn.txt now...]
    if target and tgt_dict is not None:
        tgt_dict.save(_dict_path(args.target_lang, args.destdir))

    if args.dict_only: # here! both for case1 and case2; not in for case 3
        return

    _make_all(args.source_lang, src_dict, args)
    if target:
        _make_all(args.target_lang, tgt_dict, args)

    # align the datasets if needed
    if args.align_suffix:
        _make_all_alignments(args)

    logger.info("Wrote preprocessed data to {}".format(args.destdir))
    # 2022-10-14 16:23:23 | INFO | fairseq_cli.preprocess | Wrote preprocessed data to /workspace/asr/wav2vec/fairseq/examples/wav2vec/data/librispeech/train/txt/phones
    if args.alignfile:
        _align_files(args, src_dict=src_dict, tgt_dict=tgt_dict)


def cli_main():
    parser = options.get_preprocessing_parser()
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
