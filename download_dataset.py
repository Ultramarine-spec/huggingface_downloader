import os
import argparse
from datasets import load_dataset, load_from_disk, get_dataset_config_names

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser = argparse.ArgumentParser(description="Download a dataset from huggingface hub.")
parser.add_argument("--dataset_name", "-n", type=str, default=None, required=True)
parser.add_argument("--dir", type=str, default=None, help="Local dir to save the dataset")
parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache dir")
parser.add_argument("--subset", "-sub", type=str, default=None, help="Subset of the dataset")
parser.add_argument("--all", action='store_true', help="Whether to download all subsets of the dataset")
args = parser.parse_args()

def my_load_dataset(path, name=None, cache_dir=args.cache_dir, root=args.dir):
    if name is None:
        root = os.path.join(root, path)
    else:
        root = os.path.join(root, path, name)
    if os.path.exists(root):
        raw_datasets = load_from_disk(root)
    else:
        raw_datasets = load_dataset(path=path, name=name, cache_dir=cache_dir)
        raw_datasets.save_to_disk(root)
    return raw_datasets

if args.all:
    subset_names = get_dataset_config_names(args.dataset_name)
    print(subset_names)
    for sub in subset_names:
        my_load_dataset(path=args.dataset_name, name=sub)
else:
    my_load_dataset(path=args.dataset_name, name=args.subset)



