## Efficiently Summarizing Text and Graph Encodings of Multi-Document Clusters (NAACL 2021)

This is the implementation of the paper [Efficiently Summarizing Text and Graph Encodings of Multi-Document Clusters](https://www.aclweb.org/anthology/2021.naacl-main.380.pdf).

## Pre-installations
1. This code is tested using Python 3.6, Pytorch 1.4, and CUDA 10.1

2. Install Apex:
      - ```cd ~; git clone https://github.com/NVIDIA/apex```
      - ```cd apex```
      - ```pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" ./```
3. Install nccl:
      - ```cd ~; git clone https://github.com/NVIDIA/nccl.git```
      - ```cd nccl```
      - ```make -j10 src.build CUDA_HOME=/usr/local/cuda-10.1```
      - ```sudo apt install build-essential devscripts debhelper fakeroot```
      - ```make -j10 pkg.debian.build CUDA_HOME=/usr/local/cuda-10.1```
      - ```sudo apt install ./build/pkg/deb/*.deb```
4. Setup fairSeq and import some files:
      - ```cd ~```
      - ```pip uninstall -y enum34``` # Prevent AttributeError: module 'enum' has no attribute 'IntFlag'
      - ```git clone --branch v0.9.0 https://github.com/pytorch/fairseq```
      - ```mkdir ~/fairseq/models```
      - ```cd ~/fairseq/models```
      - ```wget 'https://dl.fbaipublicfiles.com/fairseq/models/bart.large.tar.gz'```
      - ```tar -xzf bart.large.tar.gz```
      - ```cd ~/fairseq```
      - ```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'```
      - ```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'```
      - ```wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'```
      - Download `encoder-updated.json` file from `https://github.com/amazon-research/BartGraphSumm/blob/main/data/encoder-updated.json` and put it under `~/fairseq`
5. Install NLTK and Spacy:
      - ```pip install nltk spacy more_itertools```
      - ```python -m spacy download en_core_web_sm```
      - ```python -m nltk.downloader stopwords```
      - ```python -m nltk.downloader punkt```
      
7. For ROUGE:
      - ```sudo apt-get install -y cpanminus```
      - ```cpanm —force XML::Parser```
      - ```cd ~```
      - ```pip install -U git+https://github.com/pltrdy/pyrouge```
      - ```git clone https://github.com/pltrdy/files2rouge.git```
      - ```cd files2rouge```
      - ```python setup_rouge.py```
      - ```python setup.py install```
      - ```pyrouge_set_rouge_path ~/.files2rouge```
8. Get PreSumm (for fast parallel ROUGE impl — note that this does not split summaries into sentences and therefore gives worse ROUGE-L scores than files2rouge)
      - ```cd ~```
      - ```git clone https://github.com/nlpyang/PreSumm.git```   
9. Other:
      - ```mkdir -p ~/results```
      - ```pip install rouge```

## Prepare Multi-News Data

1. ```cd ~; mkdir data; cd data```
2. multi-news-500 (Preprocessed and truncated data):
      - Get it from [original source](https://drive.google.com/drive/folders/1qqSnxiaNVEctgiz2g-Wd3a9kwWuwMA07) and rename the folder as `multi-news-500`; Also, rename xxx.txt.src.yyy as xxx.source and xxx.txt.tgt.yyy as xxx.target

3. multi-news-full-clean (Preprocessed but not truncated):
      - Get it from [original source](https://drive.google.com/open?id=1qZ3zJBv0zrUy4HVWxnx33IsrHGimXLPy) and rename the folder as `multi-news-full-clean`; Also, rename the files inside this folder as follows: xxx.txt.src as xxx.source and xxx.txt.tgt as xxx.target

4. multi-news-full-raw (not processed and not truncated) -- This is only needed for graph construction in BART-Long-Graph models.
      - Get it from [original source](https://drive.google.com/drive/folders/1uDarzpu2HFc-vjXNJCRv2NIHzakpSGOw) and rename the folder as `multi-news-full-raw`; Also, rename the files inside this folder as follows: xxx.src as xxx.source and xxx.tgt as xxx.target

## Code Setup
- ```cd ~; git clone git@github.com:amazon-research/BartGraphSumm.git```
- ```cd ~/BartGraphSumm/src/fairseq```
- ```pip install --editable .```
- ```cd ../```

## Train and Evaluate

### BART baseline
Try the following command to train and evaluate the BART baseline model on Multi-News-500 dataset
```
cd ~/BartGraphSumm/src
make -f bart-large.mk TASK=~/data/multi-news-500 OUTPUT_DIR=~/results/bart-large-multinews-model1 rouge
```

The ROUGE F1 scores (R1/R2/RL) can be found at
```
cat ~/results/bart-large-multinews-model1/test.rouge-stdout | grep ">"
>> ROUGE-F(1/2/3/l): 49.22/18.88/23.88
```
The scores correspond to the numbers from BART (input length=500) in Table 4 in the paper.

### BART-Long
1. Create a longformer model
      - ```cd ~; cp -r ~/fairseq/models/bart.large ~/fairseq/models/bart.large.long```
      - ```cd ~/BartGraphSumm/src; python convert_model_to_long.py --input_model_path ~/fairseq/models/bart.large/model.pt --output_model_path ~/fairseq/models/bart.large.long/model.pt```
2. Create data of length 500 tokens:
      - ```cd ~/BartGraphSumm/src```
      - ```python prepare_data.py --data_path ~/data/multi-news-full-clean --output_path ~/data/multi-news-500-sentmarkers --max_length 500 --sentence_level_markers```
```
make -f bart-large-long.mk TASK=~/data/multi-news-500-sentmarkers MAX_TOKENS=1024 OUTPUT_DIR=~/results/bart-large-multinews-model2 rouge
```

The ROUGE F1 scores (R1/R2/RL) can be found at
```
cat ~/results/bart-large-multinews-model2/test.rouge-stdout | grep ">"
>> ROUGE-F(1/2/3/l): 48.54/18.56/23.78
```
The scores correspond to the numbers from Bart-Long in Table 1 in the paper.

### BART-Long-Graph
1. Create BART-Long model with additional encoder for encoding the graph information
      - ```cp -r ~/fairseq/models/bart.large ~/fairseq/models/bart.large.long.graph.linear```
      - ```cd ~/BartGraphSumm/src; python convert_model_to_long.py --input_model_path ~/fairseq/models/bart.large/model.pt --output_model_path ~/fairseq/models/bart.large.long.graph.linear/model.pt --linear_graph```
2. Create the graph data and its linearized form
      - Create a new virtual env with pytorch 1.5 --> our graph construction code relies on latest allennlp modules which has pytorch 1.5 requirements. So, please follow the below setups in creating a new virtual environment
      - ```sudo apt-get install virtualenv```
      - ```cd ~/; virtualenv -p python3.6 graph_env; source graph_env/bin/activate```
      - ```pip install numpy allennlp==1.0.0 allennlp_models==1.0.0 networkx==2.4 matplotlib==3.3.0```
      - ```cd ~/BartGraphSumm/src```
      - ```python graph_construction.py --data_path ~/data/multi-news-full-raw --output_path ~/data/multi-news-full-graph --split train```
      - ```python graph_construction.py --data_path ~/data/multi-news-full-raw --output_path ~/data/multi-news-full-graph --split val```
      - ```python graph_construction.py --data_path ~/data/multi-news-full-raw --output_path ~/data/multi-news-full-graph --split test```
      - Load back the Pytorch 1.4 environment
      - Now create the data by concatenating plan text input with graph information for each sample in the multi-news dataset: ```python prepare_data.py --data_path ~/data/multi-news-full-clean --output_path ~/data/multi-news-500-500 --max_length 500 --mode standard_with_graph_knowledge --graph_data_path ~/data/multi-news-full-graph --sentence_level_markers```

```
make -f bart-large-graph-linear.mk TASK=~/data/multi-news-500-500 MAX_TOKENS=1500 MAX_EPOCH=8 OUTPUT_DIR=~/results/bart-large-multinews-model3 rouge
```
The ROUGE F1 scores (R1/R2/RL) can be found at
```
cat ~/results/bart-large-multinews-model3/test.rouge-stdout | grep ">"
>> ROUGE-F(1/2/3/l): 49.03/19.04/24.04
```
The scores correspond to the numbers from Bart-Long-Graph (500 tokens graph text) in Table 1 in the paper.

To create the data with 1000 tokens for graph information: ```python prepare_data.py --data_path ~/data/multi-news-full-clean --output_path ~/data/multi-news-500-1000 --max_length 1000 --mode standard_with_graph_knowledge --graph_data_path ~/data/multi-news-full-graph --sentence_level_markers```

and train the model: 
```
make -f bart-large-graph-linear.mk TASK=~/data/multi-news-500-1000 MAX_TOKENS=2500 MAX_EPOCH=8 LR=3e-05 OUTPUT_DIR=~/results/bart-large-multinews-model4 rouge
```

The ROUGE F1 scores (R1/R2/RL) can be found at
```
cat ~/results/bart-large-multinews-model4/test.rouge-stdout | grep ">"
>> ROUGE-F(1/2/3/l): 49.24/18.99/23.97
```
The scores correspond to the numbers from Bart-Long-Graph (1000 tokens graph text) in Table 1 in the paper.


## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.

## Reference

If you find this code helpful, please consider citing the following paper:

    @inproceedings{pasunuru2021efficient,
        title={Efficiently Summarizing Text and Graph Encodings of Multi-Document Clusters},
        author={Pasunuru, Ramakanth and Liu, Mengwen and Bansal, Mohit and Ravi, Sujith and Dreyer, Markus},
        booktitle={NAACL},
        year={2021}
    }