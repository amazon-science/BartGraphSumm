#!/usr/bin/env python
##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

import argparse
import os
from nltk.tokenize import sent_tokenize
from graph_utils import cluster_docs
from multiprocessing import Pool
from tqdm import tqdm
import json
import numpy as np
import random
import math
from rouge import Rouge
#import ipdb

random.seed(111)

SPLITs = ['train', 'val', 'test']
GRAPH_SENT_LIMIT = 100
RANKING_OFFSET = 2

SEPARATER_TAG = "</s>"


def get_graph_info(data):
    srcs, graph_encoding_method, similarity_metric, worker_id = data
    src_graph_info = []
    for i, src in enumerate(srcs):
        docs = src
        info = cluster_docs(docs, graph_method=graph_encoding_method, similarity_metric=similarity_metric, threshold=None)
        info["sim_mat"] = None # saving disk space when saving
        if i%100==0 and i>0:
            print(f"Processed {i} samples with worker-{worker_id}")
        src_graph_info.append(info)

    return src_graph_info


def read_tgt(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the target file: {filename}")
    tgts = []
    with open(filename, "r") as f:
        for line in f.read().splitlines():
            tgts.append(line.strip())
    
    print("Printing a sample from this file")
    print(f"{tgts[0][:50]}[...]")

    return tgts

 
def read_src(filename):
    """ module to read the target data and 
    send it as list of tgts"""
    print(f"Reading the source file: {filename}")
    srcs = []
    with open(filename, "r") as f:
        for line in f.read().splitlines():
            line = " ".join(line.split())
            docs = line.split("story_separator_special_tag")  
            docs = [d.strip() for d in docs]
            srcs.append(docs)

    print("Printing a sample from this file")
    print(f"Total Documents in this sample: {len(srcs[0])}")
    print("#"*20)
    for d in srcs[0]:
        print(f"{d[:50]}[...]")
    print("#"*20)

    return srcs

def read_graph(filename):
    """ module to read the graph data and 
    send it as list of dictionaries"""
    print(f"Reading the graph file: {filename}")
    srcs = []
    data = [json.loads(d)["cluster"] for d in open(filename)]
    print(f"Total graphs in this file: {len(data)}")
    print("Printing a sample from this file")
    print("#"*20)
    print(f"{data[0]}")
    print("#"*20)

    return data


def truncate(line, separator_tag="story_separator_special_tag", total_words=500):
    line_word_split = line.split()
    if len(line_word_split) < total_words:
        return line
    else:
        sources_split = line.split(separator_tag)
        # previous dataset had separator at the end of each example
        if sources_split[-1] == "":
            del sources_split[-1]
        num_sources = len(sources_split)
        words_ar = [source.split() for source in sources_split]
        num_words_ar = [len(words) for words in words_ar]
        #logging.debug(f"initial number of words: {str(num_words_ar)}")
        per_source_count = math.floor(total_words / num_sources)
        total_ar = [0] * num_sources
        total = 0
        done = {}
        while total < total_words and len(done) < len(num_words_ar):
            # e.g. total=499 and still trying to add -- just add from the first doc which isn't done
            if per_source_count == 0:
                for index, x in enumerate(total_ar):
                    if index not in done:
                        total_ar[index] += total_words - total
                        break
                break
            min_amount = min(min([x for x in num_words_ar if x > 0]), per_source_count)
            total_ar = [x + min_amount if index not in done else x for index, x in enumerate(total_ar)]
            for index, val in enumerate(num_words_ar):
                if val == min_amount:
                    done[index] = True
            num_words_ar = [x - min_amount for x in num_words_ar]
            total = sum(total_ar)
            if len(done) == len(num_words_ar):
                break
            per_source_count = math.floor((total_words - total) / (len(num_words_ar) - len(done))) 
        final_words_ar = []
        for count_words, words in enumerate(words_ar):
            cur_string = " ".join(words[:total_ar[count_words]])
            final_words_ar.append(cur_string)
        final_str = (" " + separator_tag + " ").join(final_words_ar).strip()
        return final_str




def create_graphs(args):
    """ Module to create graph representations for each Multi-News
    example. """
    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)
    for split in SPLITs:
        print(f"Processing Split: {split}")
        srcs = read_src(os.path.join(args.data_path, f"{split}.source"))

        if os.path.exists(os.path.join(args.output_path, f"{split}.jsonl")) and args.start_chunk_index==0:
            if args.overwrite:
                os.remove(os.path.join(args.output_path, f"{split}.jsonl"))
            else:
                raise Exception("set overwrite argument to remove already existing files")
        f_write = open(os.path.join(args.output_path, f"{split}.jsonl"), "a")

        print(f"Getting Graph encoding information for mode: {args.graph_encoding}-{args.similarity_metric}...")
        for ind, chunk in enumerate(np.array_split(srcs, args.num_batches)[args.start_chunk_index:], start=args.start_chunk_index):
            chunk = chunk.tolist()
            print(f"Processing chunk:{ind+1} out of {args.num_batches} chunks")
            if args.num_workers>1:
                print(f"Running {args.num_workers} parallel threads.")
                srcs_per_thread = np.array_split(chunk, args.num_workers)
                src_graph_info = [] 
                args_list = []
                for worker_id in range(0, min(args.num_workers,len(srcs_per_thread))):
                    args_list.append((srcs_per_thread[worker_id].tolist(),args.graph_encoding, args.similarity_metric, worker_id))
                
                pool = Pool(processes=min(args.num_workers,len(srcs_per_thread)))
                result = pool.map(get_graph_info, args_list)
                for r in result:
                    src_graph_info = src_graph_info + r
            else:
                src_graph_info = get_graph_info((chunk, args.graph_encoding, 0))

            for info in src_graph_info:
                json.dump(info, f_write)
                f_write.write("\n")

            f_write.flush()

        f_write.close()


def parse_graph_to_special_linearization(gstr):
    """ Module to convert already linearized graph into
    a special linearization"""
    sents = []
    clusters = gstr.split("<sub>")
    for cluster in clusters:
        tuples = cluster.split("<obj>")
        sub = tuples[0].strip()
        for each in tuples[1:]:
            each = each.split("<pred>")
            obj = each[0].strip()
            pred = each[1].strip()
            if "<cat>" in pred:
                pred = pred.split(" <cat> ")
                for p in pred:
                    sents.append(f"{sub} <mask> {p} <mask> {obj} <mask>")
            else:
                sents.append(f"{sub} <mask> {pred} <mask> {obj} <mask>")

    return sents
                    



def create_data_with_graph_knowledge(args):
    """ Module to combine append extra knowledge """
    assert args.sentence_level_markers

    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)
    for split in SPLITs:
        print(f"Processing Split: {split}")
        srcs = read_src(os.path.join(args.data_path, f"{split}.source"))
        tgts = read_tgt(os.path.join(args.data_path, f"{split}.target"))


        graph_data = read_tgt(os.path.join(args.graph_data_path, f"{split}.graph"))


        # Truncate the source
        new_srcs = []
        for src in srcs:
            src = " story_separator_special_tag ".join(src) 
            src = truncate(src, total_words=args.max_length)
            src = src.split(" story_separator_special_tag ")
            new_srcs.append(src)

        f_src = os.path.join(args.output_path, f"{split}.source")
        f_tgt = os.path.join(args.output_path, f"{split}.target")


        srcs = [" ".join(src) for src in new_srcs] 
        srcs = [sent_tokenize(src) for src in srcs]
        srcs = [f" {SEPARATER_TAG} ".join(src) for src in srcs]

        special_codes = []
        empty_codes = {}
        with open("/home/ubuntu/fairseq/dict.txt", "r") as f:
            for line in f.read().splitlines():
                try:
                    line = line.split()
                    int_code = int(line[0])
                    if int(line[1])==0:
                        empty_codes[int_code] = ""
                except:
                    continue
        # Read the encoder.json word to code mapping
        with open("/home/ubuntu/fairseq/encoder.json", "r") as f:
            encoder = json.load(f)
            for k,v in encoder.items():
                if empty_codes.get(v) is not None:
                    special_codes.append(k)
        print(f"All the empty_codes: {empty_codes}")
        print(f"All the special codes: {special_codes}")


        # Append the graph data:
        new_srcs = []
        for ind, src in enumerate(srcs):
            
            if args.special_linearize:
                gstr = graph_data[ind].lower()
                gstr_list = parse_graph_to_special_linearization(gstr)
                gstr = " </s> ".join(gstr_list)
                gstr = " ".join(gstr.split()[:args.max_length + len(gstr_list)])
                new_src = src + " " + gstr
            else:
                gstr = " ".join(graph_data[ind].split()[:args.max_length])
                gstr = gstr.lower()
                gstr = gstr.replace("<sub>", "</s> <sub>")
                gstr = gstr.replace("<sub>", special_codes[0])
                gstr = gstr.replace("<obj>", special_codes[1])
                gstr = gstr.replace("<pred>", special_codes[2])
                gstr = gstr.replace("<cat>", special_codes[3])
                new_src = src + " " + gstr
                #new_src = gstr
            new_srcs.append(new_src)

        srcs = new_srcs

        with open(f_src, "w") as f:
            f.write("\n".join(srcs))
            f.flush()
            f.close()

        with open(f_tgt, "w") as f:
            f.write("\n".join(tgts))
            f.flush()
            f.close()





def create_data(args):
    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)
    for split in SPLITs:
        print(f"Processing Split: {split}")
        srcs = read_src(os.path.join(args.data_path, f"{split}.source"))
        tgts = read_tgt(os.path.join(args.data_path, f"{split}.target"))


        if args.graph_encoding:
            graph_info = read_graph(os.path.join(args.graph_data_path, f"{split}.jsonl"))

        if args.shuffle_sentences:
            print("shuffling the sentences within a document")
            new_srcs = []
            for ind, src in enumerate(srcs):
                new_docs = []
                for doc in src:
                    doc_sents = sent_tokenize(doc)
                    random.shuffle(doc_sents)
                    new_docs.append(" ".join(doc_sents))
                new_srcs.append(new_docs)
                if ind==0:
                    print(f"Sample original src:::", srcs[0])
                    print(f"Sample sentence shuffled src:::", new_srcs[0])

            srcs = new_srcs
        # Truncate the source
        new_srcs = []
        for src in srcs:
            src = " story_separator_special_tag ".join(src) 
            src = truncate(src, total_words=args.max_length)
            src = src.split(" story_separator_special_tag ")
            new_srcs.append(src)

        f_src = os.path.join(args.output_path, f"{split}.source")
        f_tgt = os.path.join(args.output_path, f"{split}.target")

        if args.sentence_level_markers:
            if args.graph_encoding:
                new_srcs_g = []
                for index, src in enumerate(new_srcs):
                    scores_list = [v["score"] for k,v in graph_info[index].items()]
                    threshold1 = np.quantile(np.array([scores_list]), 0.33)
                    threshold2 = np.quantile(np.array([scores_list]), 0.67)
                    new_docs = []
                    for i, doc in enumerate(src):
                        new_doc = []
                        for j, sent in enumerate(sent_tokenize(doc)):
                            if j < GRAPH_SENT_LIMIT:
                                id_ = "d{}_s{}".format(i,j)
                                score = graph_info[index][id_]["score"]
                                if score>threshold2:
                                    label = "high"
                                elif score>threshold1:
                                    label = "medium"
                                else:
                                    label = "low"
                                new_doc.append(sent + f" graph score is {label} {SEPARATER_TAG}")
                            else:
                                new_doc.append(sent + f" {SEPARATER_TAG}")
                        new_docs.append(" ".join(new_doc))
                    new_srcs_g.append(" ".join(new_docs))
                srcs = new_srcs_g
            else:      
                #TODO: right now the joining of docs is bad as src tokenize gets wrong
                srcs = [" ".join(src) for src in new_srcs] 
                srcs = [sent_tokenize(src) for src in srcs]
                srcs = [f" {SEPARATER_TAG} ".join(src) for src in srcs]

        else:
            srcs = [f" {SEPARATER_TAG} ".join(src) for src in new_srcs]


        with open(f_src, "w") as f:
            f.write("\n".join(srcs))
            f.flush()
            f.close()

        with open(f_tgt, "w") as f:
            f.write("\n".join(tgts))
            f.flush()
            f.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/data/multi-news-full-clean')
    parser.add_argument('--input_path', type=str, default='')
    parser.add_argument('--graph_data_path', type=str, default='/home/ubuntu/data/multi-news-full-clean-graph-pagerank-tfidf')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/data/multi-news-500')
    parser.add_argument('--max_length', type=int, default=500)
    parser.add_argument('--sentence_level_markers', action='store_true', default=False)
    parser.add_argument('--graph_encoding', type=str, default='', help="if not empty, \
                        added some graph info to the inputs in textual form")
    parser.add_argument('--similarity_metric', type=str, default='tfidf', help="choose option like rouge-1/2/l or tfidf")
    parser.add_argument('--num_workers', type=int, default=1, help="set this number of multiprocessing threads")
    parser.add_argument('--num_batches', type=int, default=1, help="set this number for processing/saving outputs in chunks")
    parser.add_argument('--overwrite', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default="standard")
    parser.add_argument('--shuffle_sentences', action='store_true', default=False)
    parser.add_argument('--start_chunk_index', type=int, default=0)
    parser.add_argument('--special_linearize', action='store_true', default=False)

    args = parser.parse_args()
    if args.mode == "create_graphs":
        create_graphs(args)
    elif args.mode == "standard_with_graph_knowledge":
        create_data_with_graph_knowledge(args)
    else:
        create_data(args)

    
