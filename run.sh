RUN_ID=baseline_1000_sp1024 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
ITERATIONS=1000 \
torchrun --standalone --nproc_per_node=4 train_gpt.py
# CUDA_VISIBLE_DEVICES=3 \