clear
export HF_ENDPOINT=https://hf-mirror.com
# export HF_HUB_ENABLE_HF_TRANSFER=1

model_name=bert-base-uncased
dir=/data/jhchen/huggingface/pretrained_model/
cache_dir=/data/jhchen/.cache/huggingface/pretrained_model/

python download.py -m ${model_name} --dir ${dir} --cache_dir ${cache_dir}