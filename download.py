import os
import argparse
from huggingface_hub import snapshot_download

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

parser = argparse.ArgumentParser(description="Download a transformers model from huggingface hub.")
parser.add_argument("--model_name", "-m", type=str, default="Huggingface model name", required=True)
parser.add_argument("--dir", type=str, default=None, help="Local dir to save the model")
parser.add_argument("--cache_dir", type=str, default=None, help="Huggingface cache dir")
args = parser.parse_args()

if args.dir is not None:
    args.dir = os.path.join(args.dir, args.model_name)
os.makedirs(args.dir, exist_ok=True)

snapshot_download(repo_id=args.model_name, 
                  ignore_patterns=["*.h5", "*.ot", "*.msgpack", "*.safetensors", "*.onnx"],
                  local_dir=args.dir, 
                  local_dir_use_symlinks=False,
                  resume_download=True,
                  cache_dir=args.cache_dir)
