from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/paligemma-3b-pt-224",
    repo_type="model",
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="paligemma-3b-pt-224",
)
