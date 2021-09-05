##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

#!/usr/bin/env python

"""Decode BART model, see
https://github.com/pytorch/fairseq/blob/master/examples/bart/README.summarization.md

"""

from __future__ import print_function
import os
import sys
import argparse
import logging
import torch
from warnings import warn
from multiprocessing import Pool
from functools import partial
import more_itertools as mit
from fairseq.models.bart import BARTModel


# If you have custom architectures or tasks you've used in fairseq's
# train.py (using --user-dir) you can specify it here as well. Call
# the module fairseq_extensions and add enclosing directory to
# PYTHONPATH.
try:
    import fairseq_extensions
except:
    warn('No fairseq extension loaded')


def run(data):
    task, model_path, source, source_ids, beam, lenpen, max_len_b, min_len, no_repeat_ngram_size, batch_size, max_input_length, gpu_id = data
    logging.info("Loading model {}".format(model_path))
    model_dirname, model_fname = os.path.split(model_path)
    bart = BARTModel.from_pretrained(
        model_dirname,
        checkpoint_file=model_fname,
        data_name_or_path=task
    )
    bart.max_positions = max_input_length
    result = []
    with torch.cuda.device(gpu_id):
        bart.cuda()
        bart.eval()
        bart.half()
        inputs = [ (id_, s) for id_,s in zip(source_ids, source)]
        for i, batch_gen in enumerate(mit.chunked(inputs, batch_size)):
            batch_gen = list(batch_gen)
            batch_inputs = [b[1] for b in batch_gen]
            batch_ids = [b[0] for b in batch_gen]
            if len(batch_inputs):
                with torch.no_grad():
                    hypotheses_batch = bart.sample(batch_inputs, beam=beam, lenpen=lenpen, max_len_b=max_len_b, min_len=min_len,
                                                   no_repeat_ngram_size=no_repeat_ngram_size, sentence_ids=batch_ids)
                    for hypothesis in hypotheses_batch:
                        result.append(hypothesis)
    return result


def main(arguments):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help="Input", type=str)
    parser.add_argument('-o', '--output', help="Output", type=str, default="summaries.txt")
    parser.add_argument('-m', '--model', help="Model path", type=str, required=True)
    parser.add_argument('--task', help="Task name", type=str, required=True)
    parser.add_argument('--gpus', help="Number of GPUs", type=int, default=1)
    parser.add_argument('--beam', help="beam size", type=int, default=4)
    parser.add_argument('--lenpen', help="length penalty", type=float, default=2.0)
    parser.add_argument('--max_len_b', help="Max length for output", type=int, default=140)
    parser.add_argument('--min_len', help="Min length for output", type=int, default=55)
    parser.add_argument('--no_repeat_ngram_size', help="Ngram blocking", type=int, default=3)
    parser.add_argument('--batch_size', help="Batch size", type=int, default=32)
    parser.add_argument('--max_input_length', help="Maximum input length", type=int, default=1024)
    parser.add_argument('-t', '--test', help="Execute doctests and exit", action='store_true')
    parser.add_argument('-d', '--debug', help="Print debugging statements",
                        action="store_const", dest="loglevel", const=logging.DEBUG, default=logging.WARNING)
    parser.add_argument('-v', '--verbose', help="Verbose output",
                        action="store_const", dest="loglevel", const=logging.INFO)

    args = parser.parse_args(arguments)
    logging.basicConfig(level=args.loglevel)

    input = open(args.input, 'r', encoding="utf-8")
    output = open(args.output, 'w', encoding="utf-8")

    input_text = [line for line in input]
    pool = Pool(args.gpus)
    args_list = []
    counter = 0
    for i, input_split in enumerate(mit.divide(args.gpus, input_text)):
        input_split = list(input_split)
        input_ids = list(range(counter, counter+len(input_split)))
        counter += len(input_split)
        args_list.append((args.task, args.model, input_split, input_ids,
                          args.beam, args.lenpen, args.max_len_b, args.min_len,
                          args.no_repeat_ngram_size, args.batch_size, args.max_input_length, i))
    results = pool.map(run, args_list)
    for r in results:
        for line in r:
            print(line, file=output)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
