
# export HF_ENDPOINT="https://hf-mirror.com"
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="Iceclear/StableSR", filename="stablesr_768v_000139.ckpt",local_dir='./checkpoints')

# hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")