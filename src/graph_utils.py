##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

#!/usr/bin/env python

import numpy as np
from nltk.tokenize import sent_tokenize
import networkx as nx
from scipy.spatial import distance
from scipy.linalg import norm
from rouge import Rouge
import time


def compute_tf(word_dict, bow):
    tf_dict = {}
    bow_count = len(bow)

    for word, count in word_dict.items():
        if bow_count == 0: # to handle weird cases where bow is empty
            tf_dict[word] = count
        else:
            tf_dict[word] = count/float(bow_count)
    return tf_dict


def compute_idf(doc_list):
    import math
    idf_dict = {}
    N = len(doc_list)
    
    idf_dict = dict.fromkeys(doc_list[0].keys(), 0)
    for doc in doc_list:
        for word, val in doc.items():
            if val > 0:
                idf_dict[word] += 1
    
    for word, val in idf_dict.items():
        idf_dict[word] = math.log10(N / float(val))
        
    return idf_dict


def compute_tfidf(tf_bow, idfs):
    tfidf = {}
    for word, val in tf_bow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

def compute_base_features(docs):
    #NOTE: Assuming that docs are tokenized.
    word_set = set(" ".join(docs).split())
    word_set_list = list(word_set)
    total_num_words = len(word_set_list)
    word_index_mapping = {}
    for idx, word in enumerate(word_set_list):
        word_index_mapping[word] = idx
    bow = []
    word_dict = []
    for doc in docs:
        b = doc.split()
        bow.append(b)
        wdict = dict.fromkeys(word_set, 0)
        for word in b:
            wdict[word] += 1
        word_dict.append(wdict)

    tfs_bow = []
    for i in range(len(docs)):
        tfs_bow.append(compute_tf(word_dict[i], bow[i]))

    idfs = compute_idf(word_dict)
    tfidfs = []
    for i in range(len(docs)):
        tfidfs.append(compute_tfidf(tfs_bow[i], idfs))


    return tfidfs, word_index_mapping, total_num_words

    

def cluster_docs(docs, 
                graph_method, 
                similarity_metric="tfidf", 
                threshold=None,
                neighbour_weights=None,
                per_doc_sent_limit=100,
                time_analysis=False
    ):
    """ Module to cluster the documents and assign
    rank to each sentence based on the importance of
    that sentence in the cluster. This importance is
    calculated based on the defined `graph_method'. 
    """
    program_start_time = time.time()
    if similarity_metric=="tfidf":
        tfidfs, word_index_mapping, total_num_words = compute_base_features(docs)
    elif "rouge" in similarity_metric:
        rouge = Rouge()
    
    # transform the docs into cluster
    start_time = time.time()
    cluster = []
    for i, doc in enumerate(docs):
        for j,sent in enumerate(sent_tokenize(doc)[:per_doc_sent_limit]):
            info = {}
            if similarity_metric == "tfidf":
                vector = np.zeros(total_num_words)
                for word in sent.split():
                    try:
                        vector[word_index_mapping[word]]=tfidfs[i][word]
                    except:
                        print("Error in the key mapping; ignoring for now")
            else:
                vector = None

            info["id"] = "d{}_s{}".format(i,j)
            info["sentence"] = sent
            info["vector"] = vector
            cluster.append(info)
    
    if time_analysis:
        print(f"execution time for transform module is {time.time()-start_time}")

    start_time = time.time()
    
    # Create an Adjacency Marix
    sim_mat = np.zeros((len(cluster),len(cluster)))
                
    for i in range(len(cluster)):
        for j in range(len(cluster)):
            if neighbour_weights is None:
                tmp_counter = j
            else:
                tmp_counter = j-1
            if i<tmp_counter:
                if similarity_metric == "tfidf":
                    if norm(cluster[i]["vector"]) < 1e-20 or norm(cluster[j]["vector"])< 1e-20: # special case were vector is super spare! sent has may be one word
                        sim_mat[i,j] = 0
                    else:
                        sim_mat[i,j] = -distance.cosine(cluster[i]["vector"], cluster[j]["vector"])+1 # use 1-cosine distane to calculate similarity
                else:
                    try:
                        if len(cluster[i]["sentence"].split())==1 or \
                            len(cluster[j]["sentence"].split())==1:
                            sim_mat[i,j] = 0
                        else:
                            sim_mat[i,j] = rouge.get_scores(cluster[i]["sentence"], cluster[j]["sentence"])[0][similarity_metric]["f"]
                    except:
                        print(f"Rouge exception for pair sample: {cluster[i]['sentence']} ||| {cluster[j]['sentence']}")
                        sim_mat[i][j] = 0

            elif i>tmp_counter:
                sim_mat[i,j] = sim_mat[j,i]
            else:
                if i==j:
                    sim_mat[i,j] = 0 # when i==j
                else: ## adjacent nodes
                    sim_mat[i,j] = neighbour_weights
    if time_analysis:
        print(f"execution time for adjacency matrix creation is {time.time()-start_time}")

    start_time = time.time()
    # Graph score calculation
    if threshold is not None:
        sim_mat[sim_mat<threshold] = 0
    #sim_mat = np.exp(sim_mat)
    nx_graph = nx.from_numpy_array(sim_mat)
    final_graph_method = graph_method
    if graph_method=="pagerank":
        try:
            scores = nx.pagerank(nx_graph)
        except:
            print("Power Iteration failed..switching to generic..")
            scores = np.sum(sim_mat, axis=1)
            final_graph_method = "generic"
    elif graph_method=="generic":
        scores = np.sum(sim_mat, axis=1)
    else:
        raise Exception("Unknown graph_method: {}".format(graph_method))

    if time_analysis:
        print(f"execution time for graph calculation is {time.time()-start_time}")

    # Data format
    start_time = time.time()
    cluster_dict = {}
    cluster_dict["cluster"] = {}
    for i, sent_info in enumerate(cluster):
        sent_info["score"] = scores[i]
        sent_info["vector"] = None
        cluster_dict["cluster"][sent_info["id"]] = sent_info

    cluster_dict["sim_mat"] = sim_mat.tolist()
    cluster_dict["graph_method"] = final_graph_method

    if time_analysis:
        print(f"execution time for data formating module is {time.time()-start_time}")

    if time_analysis:
        print(f"execution time for whole module is {time.time()-program_start_time}")

    return cluster_dict



