import os
from huggingface_hub import login, HfApi, create_repo
hf_token = os.getenv('HF_TOKEN', '').strip()
hf_username = os.getenv('HF_USERNAME', '').strip()

if not hf_token:
    raise ValueError("HF_TOKEN secret is not set or is empty")
if not hf_username:
    raise ValueError("HF_USERNAME secret is not set or is empty")

SPACE_REPO  = f"{hf_username}/superkart-streamlit"
login(token=hf_token); api = HfApi()
try: create_repo(SPACE_REPO, repo_type="space", space_sdk="streamlit",
                 exist_ok=True, token=HF_TOKEN)
except: pass
for f in ["app.py","requirements.txt","Dockerfile"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f,
                    repo_id=SPACE_REPO, repo_type="space", token=hf_token)
    print(f"Deployed {f}")
print(f"Live: [huggingface.co](https://huggingface.co/spaces/{SPACE_REPO})")
