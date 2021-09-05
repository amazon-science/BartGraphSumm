##  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
##  SPDX-License-Identifier: MIT

.PHONY: train decode rouge info

MAKEFILE_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
CAL_ROUGE=$(HOME)/PreSumm/src/cal_rouge.py
BART_LARGE=$(HOME)/fairseq/models/bart.large.long/model.pt
OUTPUT_DIR=output
CHECKPOINT=$(OUTPUT_DIR)/checkpoint_best.pt
VOCAB_JSON=$(HOME)/fairseq/encoder-updated.json
VOCAB_BPE=$(HOME)/fairseq/vocab.bpe
DICT_PATH=$(HOME)/fairseq/dict.txt

TOTAL_NUM_UPDATES=20000  
WARMUP_UPDATES=500      
LR=2e-05
MAX_TOKENS=1500
MAX_EPOCH=5
UPDATE_FREQ=4
RESTORE_FILE=$(BART_LARGE)

CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
THREADS=32

TEST_NAME=test
DECODE_BATCH_SIZE=16
DECODE_MIN_LEN=50
DECODE_MAX_LEN=500

DEV_NAME=val


# CONFIG=$(MAKEFILE_DIR)/bart_config.mk
# include $(CONFIG)

# Good for overwriting machine-specific settings, e.g., CUDA_VISIBLE_DEVICES
CONFIG_LOCAL=$(HOME)/.bart_config.mk
-include $(CONFIG_LOCAL)

ifeq ($(DISABLE_VALIDATION), 1)
  VALIDATION_OPTS=--disable-validation
endif


$(TASK)-bin: $(TASK)
	for SPLIT in train $(DEV_NAME); do \
	  for EXT in source target; do \
	    python3.6 -m examples.roberta.multiprocessing_bpe_encoder --encoder-json $(VOCAB_JSON) --vocab-bpe $(VOCAB_BPE) --inputs "$(TASK)/$$SPLIT.$$EXT" --outputs "$(TASK)/$$SPLIT.bpe.$$EXT" --workers 60 --keep-empty; \
	  done; \
	done
	fairseq-preprocess --source-lang "source" --target-lang "target" --trainpref "$(TASK)/train.bpe" --validpref "$(TASK)/$(DEV_NAME).bpe" --destdir "$(TASK)-bin/" --workers 60 --srcdict $(DICT_PATH) --tgtdict $(DICT_PATH)

train: $(OUTPUT_DIR)/checkpoint_best.pt
	@echo Done

info: ; $(foreach v, $(filter-out $(VARS_OLD) VARS_OLD,$(.VARIABLES)), $(info $(v) = $($(v))))
	@

# Options:
# https://fairseq.readthedocs.io/en/latest/command_line_tools.html

$(OUTPUT_DIR)/checkpoint_best.pt: $(TASK)-bin
	mkdir -p $(OUTPUT_DIR)
	date > $(OUTPUT_DIR)/time-start.txt
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	  python3.6 fairseq/train.py $(TASK)-bin \
	    --log-format json \
	    --no-progress-bar \
	    --restore-file $(RESTORE_FILE) \
	    --max-tokens $(MAX_TOKENS) \
	    --max-epoch $(MAX_EPOCH) \
	    --task translation --source-lang source --target-lang target \
	    --layernorm-embedding --share-all-embeddings --share-decoder-input-output-embed \
	    --reset-optimizer --reset-dataloader --reset-meters \
	    --required-batch-size-multiple 1 \
	    --arch bart_large_long \
	    --criterion label_smoothed_cross_entropy \
	    --label-smoothing 0.1 \
	    --dropout 0.1 --attention-dropout 0.1 \
	    --weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
	    --clip-norm 0.1 \
	    --lr-scheduler polynomial_decay --lr $(LR) --total-num-update $(TOTAL_NUM_UPDATES) --warmup-updates $(WARMUP_UPDATES) \
	    --fp16 --update-freq $(UPDATE_FREQ) \
	    --skip-invalid-size-inputs-valid-test \
	    $(VALIDATION_OPTS) \
	    --save-dir $(OUTPUT_DIR)  \
	    --truncate-source \
	    --no-epoch-checkpoints \
        --skip-invalid-size-inputs-valid-test \
	  2>$(OUTPUT_DIR)/stderr.txt | tee $(OUTPUT_DIR)/stdout.txt
	date > $(OUTPUT_DIR)/time-end.txt

decode: $(OUTPUT_DIR)/$(TEST_NAME).decoded
	@echo Done

rouge: $(OUTPUT_DIR)/$(TEST_NAME).rouge-stdout
	@echo Done

$(OUTPUT_DIR)/$(TEST_NAME).decoded: $(CHECKPOINT) $(TASK)/$(TEST_NAME).source
	python3.6 $(MAKEFILE_DIR)/bart_decode_parallel.py --gpus=$(NUM_GPUS) --batch_size=$(DECODE_BATCH_SIZE) --max_input_length $(MAX_TOKENS) \
	  --model=$(CHECKPOINT) --min_len=$(DECODE_MIN_LEN) --max_len_b=$(DECODE_MAX_LEN) --task=$(TASK)-bin -o $@ $(TASK)/$(TEST_NAME).source

$(OUTPUT_DIR)/$(TEST_NAME).rouge-stdout: $(OUTPUT_DIR)/$(TEST_NAME).decoded
	python3.6 $(CAL_ROUGE) -p $(THREADS) -r $(TASK)/$(TEST_NAME).target -c $(OUTPUT_DIR)/$(TEST_NAME).decoded 2>$(OUTPUT_DIR)/$(TEST_NAME).rouge-stderr >$(OUTPUT_DIR)/$(TEST_NAME).rouge-stdout

