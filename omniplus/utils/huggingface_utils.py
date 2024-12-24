from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct",
    repo_type="model",
    ignore_patterns=["*.md", "*..gitattributes"],
    local_dir="Phi-3-mini-4k-instruct",
)
