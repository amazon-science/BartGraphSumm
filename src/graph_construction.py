##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT


import numpy as np
from numpy import dot
from numpy.linalg import norm
import allennlp
import argparse
import spacy
import math
import os
from spacy.lang.en import English
nlp = English()
nlp.add_pipe(nlp.create_pipe('sentencizer'))
from allennlp.predictors.predictor import Predictor
import allennlp_models.coref
import allennlp_models.structured_prediction
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance
from graph_utils import compute_base_features
import time
import torch
import json
from torch.multiprocessing import Pool, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass




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


def extract_coref(predictor, document):
    print(f"document length: {len(document.split())}")
    try:
        result = predictor.predict(
            document=document
        )
        document_tokenized = result["document"]
        cluster = {}
        for c in result['clusters']:
            unique_map = " ".join(document_tokenized[c[0][0]:c[0][1]+1])
            for ind, span in enumerate(c):                
                key = "{}-{}".format(span[0],span[1])
                value = " ".join(document_tokenized[span[0]:span[1]+1])
                cluster[key] = {"unique": unique_map, "span": value}

    except:
        document_tokenized = []
        cluster = {}
        print("Error occured in document coreference resolution")


    return document_tokenized, cluster


def parser_oie(result, tokens, offset):
    # Result format is same as IOE extaction output
    sample = {}
    start = None
    end = None
    key = None
    for ind,tag in enumerate(result["tags"]):
        if tag == "O":
            if start is not None:
                if end is None:
                    sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                else:
                    sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                start = None
                end = None
                key = None
        else:
            if tag[:1] == "B":
                if start is not None:
                    if end is None:
                        sample[key] = {"span": " ".join(tokens[start:start+1]), "index": [start+offset, start+offset]}
                    else:
                        sample[key] = {"span": " ".join(tokens[start:end+1]), "index": [start+offset, end+offset]}
                end = None
                start = ind
                key = tag[2:]
            else:
                end = ind
    return sample
        

def extract_oie(predictor, document):
    counter = 0
    ioes = []
    json_input = [{"sentence": sent.text } for sent in nlp(document).sents]
    result_list = predictor.predict_batch_json(
        json_input
    )
    for result in result_list:
        tokens = result["words"]
        for r in result["verbs"]:
            ioe_sample = parser_oie(r, tokens, counter)
            #print(ioe_sample)
            if ioe_sample.get("V") is not None and \
                ioe_sample.get("ARG0") is not None and \
                ioe_sample.get("ARG1") is not None:
                ioes.append(ioe_sample)

        counter += len(tokens)

    return ioes


def similar_match(
        text, span_list, 
        text_vec=None, 
        span_vec_list=None,
        threshold=0.5
    ):

    def cosine_similarity(a, b):
        numerator = dot(a,b)
        if math.isclose(numerator, 0):
            return 0
        else:
            demoninator = norm(a) * norm(b)
            return numerator/demoninator

    match_index = -1 
    for ind, span in enumerate(span_list):
        if span==text:
            match_index=ind
            break
    if match_index != -1:
        return match_index
    elif text_vec is not None and len(span_list)!=0:
        sim_scores = []
        #ipdb.set_trace()
        for vec in span_vec_list:
            sim_scores.append(cosine_similarity(vec, text_vec))
        sim_scores = np.array(sim_scores)
        max_ind = np.argmax(sim_scores)
        if sim_scores[max_ind]>threshold:
            return max_ind
        else:
            return -1
    else:
        return -1


def bfs_linearize(graph, root_node, directed_edges):
    visited = [root_node]
    queue = [root_node]
    linearize_graph = []
    while queue:
        s = queue.pop(0)
        ltext = [] 
        for n in graph.neighbors(s): # TODO: can make better with order or neighbors based on their weight
            if n not in visited:
                #if directed_edges.get(f"{s}-{n}") is not None:
                    ltext += [f"<obj> {graph.nodes[n]['text']}"]
                    predicates = "<pred> "+" <cat> ".join([p['text'] for p in graph[s][n]["preds"]])
                    ltext += [predicates]
                
                    visited.append(n)
                    queue.append(n)

        if len(ltext) > 0:
            ltext = [f"<sub> {graph.nodes[s]['text']}"] + ltext
            linearize_graph.append(" ".join(ltext))


    return linearize_graph

            
           

def linearize_graph(graph_info):
    # Get connected subgraphs
    graph = graph_info['graph']
    directed_edges = graph_info['directed_edges']
    sub_graphs = []
    counter = 0
    for ind, c in enumerate(nx.connected_components(graph)):
        g = graph.subgraph(c)
        counter += 1
        #print(f"Connected subgraph-{ind}: {g.nodes.data()}")
        if len(list(c))>2:# excluding graphs with less than 3 nodes
            sub_graphs.append(g)
    #print(f"Total Subgraphs: {counter}")
    #print(f"Total Subgraphs after filter: {len(sub_graphs)}")
    # Sort these subgraphs in the order of their count of nodes
    sub_graphs_sorted = sorted(sub_graphs, key=lambda x: len(x), reverse=True)
    graph_linearization = []
    for g in sub_graphs_sorted:
        g_list = list(g.nodes)
        total_nodes = len(g_list)
        root_node = g_list[np.argmax(np.array([g.nodes[i]["weight"] for i in g_list]))]
        graph_linearization += bfs_linearize(g, root_node, directed_edges)

    return graph_linearization    



def construct_graph(docs, coref_predictor, oie_predictor):
    graph = nx.Graph()
    node_names_list = []
    node_vec_list = []
    node_counter = 0
    # Get Coref:
    tokenized_docs = []
    coref_info_list = []
    start_time = time.time()
    for ind, d in enumerate(docs):
        dtokens, cinfo = extract_coref(coref_predictor, d)
        tokenized_docs.append(" ".join(dtokens))
        coref_info_list.append(cinfo)
    #print(f"Time for coref extraction for all documents: {time.time()-start_time}")
 
    tfidfs_list, word_index_mapping, total_num_words = compute_base_features(tokenized_docs)
    generic_tfidfs = {} # This is to handle errors when sub_name is not in current docs tfidfs
    for tfidf in tfidfs_list:
        for k,v in tfidf.items():
            if generic_tfidfs.get(k) is None:
                generic_tfidfs[k] = v
            else:
                if v>generic_tfidfs[k]:
                    generic_tfidfs[k] = v
                
    #print(generic_tfidfs)
    oies_list = []
    directed_edges = {}
    for ind, d in enumerate(docs):
        start_time = time.time()
        # Get OIE:
        oies = extract_oie(oie_predictor, d)
        #print(f"Time for extracting oie for doc-{ind} is {time.time()-start_time}")
        coref_info = coref_info_list[ind]
        tfidfs = tfidfs_list[ind]
        #print(tfidfs)
        oies_list.append(oies)
        for x in oies:
            #print(x)
            # Node for subject / ARG0
            uid = "{}-{}".format(x["ARG0"]["index"][0], x["ARG0"]["index"][1])
            if coref_info.get(uid) is not None:
                sub_name = coref_info[uid]["unique"]
            else:
                sub_name = x["ARG0"]["span"]
            # Add vector
            sub_name_vector = np.zeros(total_num_words)
            for word in sub_name.split():
                try:
                    sub_name_vector[word_index_mapping[word]]= tfidfs[word] if tfidfs.get(word) is not None else generic_tfidfs[word]
                except:
                    print(f"Exception Occurred for word: {word}")

            sim_node_ind = similar_match(sub_name, node_names_list, sub_name_vector, node_vec_list)
            if sim_node_ind == -1:
                graph.add_node(node_counter, text=sub_name, weight=1)
                sub_index = node_counter
                node_counter += 1
                node_names_list.append(sub_name)
                node_vec_list.append(sub_name_vector)
            else:
                sub_index = sim_node_ind
                graph.nodes[sub_index]["weight"] += 1


            # Node for Object / ARG1
            uid = "{}-{}".format(x["ARG1"]["index"][0], x["ARG1"]["index"][1])
            if coref_info.get(uid) is not None:
                obj_name = coref_info[uid]["unique"]
            else:
                obj_name = x["ARG1"]["span"]
            # Add vector
            obj_name_vector = np.zeros(total_num_words)
            for word in obj_name.split():
                try:
                    obj_name_vector[word_index_mapping[word]]=tfidfs[word] if tfidfs.get(word) is not None else generic_tfidfs[word]
                except:
                    print(f"Exception Occurred for word: {word}")


            sim_node_ind = similar_match(obj_name, node_names_list, obj_name_vector, node_vec_list)
            if sim_node_ind == -1:
                graph.add_node(node_counter, text=obj_name, weight=1)
                obj_index = node_counter
                node_counter += 1
                node_names_list.append(obj_name)
                node_vec_list.append(obj_name_vector)
            else:
                obj_index = sim_node_ind
                graph.nodes[obj_index]["weight"] += 1

            # Edge info
            pred_name = x["V"]["span"]
            if graph.has_edge(sub_index, obj_index):
                sim_ind = similar_match(pred_name, [x["text"] for x in graph[sub_index][obj_index]["preds"]]) # NOTE: no vector matching 
                if sim_ind==-1:
                    graph[sub_index][obj_index]["preds"].append({"text":pred_name, "weight":1})
                else:
                    graph[sub_index][obj_index]["preds"][sim_ind]["weight"] += 1
            else:
                graph.add_edge(sub_index, obj_index, preds=[{"text":pred_name, "weight":1}])
                directed_edges[f"{sub_index}-{obj_index}"] = 1

    
    extra_info = {}
    extra_info['coref_info'] = coref_info_list
    extra_info['oie_info'] = oies_list
    extra_info['tfidfs'] = tfidfs_list
    extra_info['generic_tfidfs'] = generic_tfidfs
    extra_info['word_index_mapping'] = word_index_mapping
    extra_info['total_num_words']  = total_num_words
    graph_info = {'graph': graph, 'directed_edges': directed_edges}
    return graph_info, extra_info               
                        
                       


def main(args):
    data = []
    if not os.path.exists(args.output_path):
        print(f"Creating directory: {args.output_path}")
        os.mkdir(args.output_path)
    if args.resume:
        outfile = open(os.path.join(args.output_path, f"{args.split}.graph"), "a")
        extra_outfile = open(os.path.join(args.output_path, f"{args.split}_info.json"), "a")
    else:
        outfile = open(os.path.join(args.output_path, f"{args.split}.graph"), "w")
        extra_outfile = open(os.path.join(args.output_path, f"{args.split}_info.json"), "w")
    
    with open(os.path.join(args.data_path, f"{args.split}.source"), "r") as f:
        for line in f.read().splitlines():
            data.append(line.strip())
    #num_gpus = 1
    processed_data = []
    for sample in data:
        # Clean the sample to docs and sentences
        sample = sample[:-5] # remove the last |||||
        # remove nbsp or nbsp;
        if "nbsp;" in sample:
            sample = sample.replace("nbsp;", "")
        if "nbsp" in sample:
            sample = sample.replace("nbsp", "")
        if "amp;" in sample:
            sample = sample.replace("amp;", "")
        if "amp" in sample:
            sample = sample.replace("amp", "")
        sample = sample.replace("NEWLINE_CHAR", "")
        sample = " ".join(sample.split()) # remove white spaces
        sample = truncate(sample, separator_tag="|||||", total_words=args.max_length)
        docs = sample.split(" ||||| ")
        processed_data.append(docs)

    coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz", 
        cuda_device=args.cuda_device)
    oie_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz",
        cuda_device=args.cuda_device)
    if args.start_index == -1:
        start_index = 0
        end_index = len(processed_data)
    else:
        start_index = args.start_index
        end_index = args.end_index
    start_time = time.time()
    for ind, sample in enumerate(processed_data[start_index:end_index]):
        graph_info, extra_info = construct_graph(sample, coref_predictor, oie_predictor)
        graph_linear = linearize_graph(graph_info)
        graph_str = " ".join(graph_linear)
        outfile.write(graph_str+"\n")
        outfile.flush()
        # write the extra info
        json.dump(extra_info, extra_outfile)
        extra_outfile.write("\n")
        extra_outfile.flush()
        if ind%10==0:
            print(f"Completed till {ind+1} samples with last pool time: {time.time()-start_time}")
            start_time = time.time()

    outfile.close()
    return









	
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/home/ubuntu/data/multi-news-full-raw')
    parser.add_argument('--output_path', type=str, default='/home/ubuntu/data/multi-news-full-graph')
    parser.add_argument('--test_sample_path', type=str, default='/home/ubuntu/data/sample_graph_test.txt')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--start_index', type=int, default=-1)
    parser.add_argument('--end_index', type=int, default=-1)
    parser.add_argument('--max_length', type=int, default=1000)
    parser.add_argument('--cuda_device', type=int, default=0)
    parser.add_argument('--resume', action='store_true', default=False)
    args = parser.parse_args()

    main(args)


