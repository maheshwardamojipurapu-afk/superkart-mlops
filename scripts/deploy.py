
import os
from huggingface_hub import HfApi, login, create_repo
HF_TOKEN    = os.environ["HF_TOKEN"]
HF_USERNAME = os.environ["HF_USERNAME"]
SPACE_REPO  = f"{HF_USERNAME}/superkart-streamlit"
login(token=HF_TOKEN); api = HfApi()
try: create_repo(SPACE_REPO, repo_type="space", space_sdk="streamlit",
                 exist_ok=True, token=HF_TOKEN)
except: pass
for f in ["app.py","requirements.txt","Dockerfile"]:
    api.upload_file(path_or_fileobj=f, path_in_repo=f,
                    repo_id=SPACE_REPO, repo_type="space", token=HF_TOKEN)
    print(f"Deployed {f}")
print(f"Live: [huggingface.co](https://huggingface.co/spaces/{SPACE_REPO})")
