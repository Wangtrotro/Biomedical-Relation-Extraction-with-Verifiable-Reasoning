from huggingface_hub import snapshot_download

path = snapshot_download("microsoft/BioGPT")
print("✅ 模型已缓存路径：", path)
# /Users/shihansmac/.cache/huggingface/hub/models--microsoft--BioGPT/snapshots/eb0d815e95434dc9e3b78f464e52b899bee7d923