


# export HF_ENDPOINT="https://hf-mirror.com"
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Iceclear/StableSR",local_dir='./checkpoints')
# local_dir="/data/user/test",