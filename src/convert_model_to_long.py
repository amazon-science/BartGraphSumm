##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT


#!/usr/bin/env python

"""Convert Bart model into long, see documentation for Roberta and follow it
https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb

"""

from __future__ import print_function
import os
import sys
import argparse
import logging
import copy
import torch
from warnings import warn
from fairseq.models.bart import BARTModel
from fairseq.modules.longformer_multihead_attention import LongformerMultiheadAttention
from fairseq.modules import TransformerEncoderLayer
from fairseq.modules import LearnedPositionalEmbedding
from fairseq import checkpoint_utils

MAX_SOURCE_POSITIONS=4096
MAX_RANKING_POSITIONS=512
# If you have custom architectures or tasks you've used in fairseq's
# train.py (using --user-dir) you can specify it here as well. Call
# the module fairseq_extensions and add enclosing directory to
# PYTHONPATH.
try:
    import fairseq_extensions
except:
    warn('No fairseq extension loaded')





def create_long_model(model_path, 
                      save_model_to, 
                      attention_window, 
                      max_pos=None,
                      linear_graph=False,
):
    
    assert max_pos is None, "Not implemented yet"

    model_dirname, model_fname = os.path.split(model_path)

    print(f"Loading model from {model_path}")
    bart= BARTModel.from_pretrained(
        model_dirname,
        checkpoint_file=model_fname,
    )
    state = torch.load(model_path)

    args = state["args"]
    model = bart.model
    print("args.arch", args.arch)
    args.attention_window = getattr(args, "attention_window", 512)
    args.max_source_positions=MAX_SOURCE_POSITIONS

    # Change max positions for source positional embeddings
    print(f"Changing the max limit of source positional embeddings...")
    current_max_pos, embed_size = model.encoder.embed_positions.weight.shape
    print(f"Current max position is: {current_max_pos-2}; changing it to: {MAX_SOURCE_POSITIONS}")
    new_max_pos = MAX_SOURCE_POSITIONS + 2
    new_pos_embed = LearnedPositionalEmbedding(new_max_pos, embed_size, padding_idx=1)
    k = 2
    step = current_max_pos - 2
    while k < new_max_pos - 1:
        new_pos_embed.weight[k:(k + step)] = model.encoder.embed_positions.weight[2:]
        k += step

    # add the intial 2 position embeddings (these are reserved)
    new_pos_embed.weight[:2] = model.encoder.embed_positions.weight[:2]
    model.encoder.embed_positions = new_pos_embed 


    
    print(f"Changing the set attention layer")
    for i, layer in enumerate(model.encoder.layers):
        ## retrive values:
        embed_dim = layer.self_attn.embed_dim
        num_heads = layer.self_attn.num_heads
        dropout = layer.self_attn.dropout

        longformer_self_attn = LongformerMultiheadAttention(embed_dim,
                                                            num_heads,
                                                            dropout=dropout,
                                                            self_attention=True,
        )

        longformer_self_attn.k_proj = layer.self_attn.k_proj
        longformer_self_attn.v_proj = layer.self_attn.v_proj
        longformer_self_attn.q_proj = layer.self_attn.q_proj
        longformer_self_attn.out_proj = layer.self_attn.out_proj

        longformer_self_attn.k_proj_global = copy.deepcopy(layer.self_attn.k_proj)
        longformer_self_attn.v_proj_global = copy.deepcopy(layer.self_attn.v_proj)
        longformer_self_attn.q_proj_global = copy.deepcopy(layer.self_attn.q_proj)

        layer.self_attn = longformer_self_attn


    if linear_graph: 
        print("Adding extra encoder layer for Linear graph encoding")
        args.linear_graph = getattr(args, "linear_graph", True)
        model.encoder.graph_layer = TransformerEncoderLayer(args)

    
    print(f'Saving model to {save_model_to}')
    state["model"] = model.state_dict()
    torch.save(state, save_model_to)


def test_save_model(model_path):
    model_dirname, model_fname = os.path.split(model_path)

    print(f"Loading model from {model_path}")
    bart= BARTModel.from_pretrained(
        model_dirname,
        checkpoint_file=model_fname,
    )
    print(bart.args)
    print("Sucessfully loaded")



    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_path', type=str, default='/home/ubuntu/fairseq/models/bart.large/model.pt')
    parser.add_argument('--output_model_path', type=str, default='/home/ubuntu/fairseq/models/bart.large.long/model.pt')
    parser.add_argument('--attention_window_size', type=int, default=512)
    parser.add_argument('--linear_graph', action='store_true', default=False)
    parser.add_argument('--no_longformer', action='store_true', default=False)

    args = parser.parse_args()
    
    create_long_model(
        args.input_model_path,
        args.output_model_path,
        args.attention_window_size,
        linear_graph=args.linear_graph,
    )
    # Test model
    test_save_model(args.output_model_path)
        

