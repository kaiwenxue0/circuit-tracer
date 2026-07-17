import torch
from transformers import AutoTokenizer, AutoModel


device = 'cuda'
model = AutoModel.from_pretrained(
    '/home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/data/GSAI-ML/LLaDA-1.5.cleaned',
    torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(
    '/home/notebook/code/group/xuekaiwen/mask_diffusion/interpretable_models/data/GSAI-ML/LLaDA-1.5.cleaned')

# 打印所有权重名字和形状
for name, param in model.named_parameters():
    print(name, param.shape)
