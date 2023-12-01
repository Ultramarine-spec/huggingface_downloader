clear
export HF_ENDPOINT=https://hf-mirror.com


dataset_name=cais/mmlu
subset=all
dir=/data/jhchen/huggingface/datasets/
cache_dir=/data/jhchen/.cache/huggingface/datasets/

python download_dataset.py -n ${dataset_name} --dir ${dir} --cache_dir ${cache_dir} --subset ${subset}