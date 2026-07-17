from safetensors import safe_open

# 换成你的 safetensors 文件路径
path = "/mnt/workspace/xuekaiwen/mask_diffusion/interpretable_models/remote/sparsify/checkpoints/llama3/sparsify_llada8b_latents32768_bs256_mlp0/layers.0.mlp/sae.safetensors"

with safe_open(path, framework="pt", device="cpu") as f:
    print("Keys in file:", f.keys())
    for key in f.keys():
        tensor = f.get_tensor(key)
        print(f"{key:20s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")
