import os
from collections import namedtuple
from importlib import resources
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch import nn
from torch.autograd import Function

import circuit_tracer
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils.hf_utils import download_hf_uris, parse_hf_uri

def grad_input_via_index_select(indices, weight, grad_values):
    D = weight.size(1)
    # 取出选中的权重行：[..., K, D]
    W_sel = weight.index_select(0, indices.reshape(-1)).reshape(*indices.shape, D)
    # 加权求和到最后一维 K 上，得 [..., D]
    return (grad_values.to(weight.dtype).unsqueeze(-1) * W_sel).sum(dim=-2)

class TopKEncode(Function):
    @staticmethod
    def forward(ctx, x, W_enc, b_enc, k: int, use_activation: bool = True, act: str = "relu"):
        """
        x:      [N, D]
        q:  [D, M]
        b_enc:  [M] or None
        k:      top-k
        """
        # 线性 + 可选激活（与原 encode 一致）
        pre = x.to(W_enc.dtype) @ W_enc
        if b_enc is not None:
            pre = pre + b_enc
        if use_activation:
            if act == "relu":
                acts = F.relu(pre)
            elif act == "gelu":
                acts = F.gelu(pre)
            else:
                raise ValueError(f"Unsupported activation: {act}")
        else:
            acts = pre

        # 逐样本取 top-k
        values, indices = torch.topk(acts, k, dim=-1, sorted=False)

        # 反向需要：输入、W 的“行向量视图”（即 W^T 的行是每个单元的权重）、以及 indices
        W_T = W_enc.transpose(0, 1).contiguous()  # [M, D]
        ctx.save_for_backward(x, W_T, indices)
        ctx.M = W_T.shape[0]
        ctx.has_bias = b_enc is not None
        return values, indices

    @staticmethod
    def backward(ctx, g_values, g_indices):
        x, W_T, indices = ctx.saved_tensors
        # --- dL/dx ---
        # sum_j g_values * W_j ；用 embedding_bag 高效聚合
        g_x = grad_input_via_index_select(indices,  W_T, g_values.type_as(W_T))

        # --- dL/dW_T ---
        g_W_T = torch.zeros_like(W_T)    # [M, D]
        B, K = g_values.shape
        D = x.shape[1]
        chunk = 32
        for i in range(0, K, chunk):
            gv = g_values[:, i:i+chunk]                  # [B, C]
            idx = indices[:, i:i+chunk]                  # [B, C]
            contrib = (gv.unsqueeze(2) * x.unsqueeze(1)) # [B, C, D]
            g_W_T.index_add_(0, idx.reshape(-1), contrib.reshape(-1, D).type_as(g_W_T))

        # --- dL/db ---
        g_b = None
        if ctx.has_bias:
            g_b = torch.zeros(ctx.M, dtype=g_values.dtype, device=g_values.device)
            g_b.index_add_(0, indices.reshape(-1), g_values.reshape(-1))

        # 映回到原 W_enc 形状 [D, M]
        g_W_enc = g_W_T.transpose(0, 1).contiguous()

        # 对 非张量参数（k/use_activation/act）返回 None
        return g_x, g_W_enc, g_b, None, None, None



class SingleLayerTranscoder(nn.Module):
    d_model: int
    d_transcoder: int
    layer_idx: int
    W_enc: nn.Parameter
    W_dec: nn.Parameter
    b_enc: nn.Parameter
    b_dec: nn.Parameter
    W_skip: Optional[nn.Parameter]
    activation_function: Callable[[torch.Tensor], torch.Tensor]

    def __init__(
        self,
        d_model: int,
        d_transcoder: int,
        activation_function,
        layer_idx: int,
        k: int,
        skip_connection: bool = False,
    ):
        """Single layer transcoder implementation, adapted from the JumpReLUSAE implementation here:
        https://colab.research.google.com/drive/17dQFYUYnuKnP6OwQPH9v_GSYUW5aj-Rp

        Args:
            d_model (int): The dimension of the model.
            d_transcoder (int): The dimension of the transcoder.
            activation_function (nn.Module): The activation function.
            layer_idx (int): The layer index.
            skip_connection (bool): Whether there is a skip connection,
                as in https://arxiv.org/abs/2501.18823
        """
        super().__init__()

        self.d_model = d_model
        self.d_transcoder = d_transcoder
        self.layer_idx = layer_idx
        self.k = k

        self.W_enc = nn.Parameter(torch.zeros(d_model, d_transcoder))
        self.W_dec = nn.Parameter(torch.zeros(d_transcoder, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_transcoder))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    def encode(self, input_acts, apply_activation_function: bool = True, return_sparse: bool = True):
        if self.k == 0:
            pre = input_acts.to(self.W_enc.dtype) @ self.W_enc + (self.b_enc if self.b_enc is not None else 0)
            acts = self.activation_function(pre) if apply_activation_function else pre
            return acts, None, None
        else:
            # 假设 self.activation_function 是 ReLU/GELU 之一；传个字符串标识
            act_name = "relu" if self.activation_function is F.relu else "gelu"
            top_acts, indices = TopKEncode.apply(input_acts, self.W_enc, self.b_enc, self.k, apply_activation_function, act_name)

            if return_sparse:
                sparse_acts = torch.zeros(
                    top_acts.size(0),   # batch_size
                    top_acts.size(1),   # token_num
                    self.d_transcoder,  # d_transcoder
                    device=top_acts.device,
                    dtype=top_acts.dtype,
                )
                sparse_acts.scatter_(2, indices, top_acts)
                # print("[DEBUG]sparse_acts shape:", sparse_acts.shape)
                
                return top_acts, indices, sparse_acts
            else:
                return top_acts, indices, None

    def decode(self, acts, indices=None):
        def eager_decode(top_indices, top_acts, W_dec):
            return grad_input_via_index_select(
            top_indices, W_dec.mT, top_acts
        )  
        if indices is not None:
            y = eager_decode(indices, acts, self.W_dec.mT)
            return y + self.b_dec
        else:
            if acts.is_sparse:
                return (
                    torch.bmm(acts, self.W_dec.unsqueeze(0).expand(acts.size(0), *self.W_dec.size()))
                    + self.b_dec
                )
            else:
                return acts @ self.W_dec + self.b_dec

    def compute_skip(self, input_acts):
        if self.W_skip is not None:
            input_acts = input_acts.to(self.W_skip.device)
            return input_acts @ self.W_skip.T
        else:
            raise ValueError("Transcoder has no skip connection")

    def forward(self, input_acts):
        if self.k == 0:
            transcoder_acts, _, _ = self.encode(input_acts)
            decoded = self.decode(transcoder_acts)
            decoded = decoded.detach()
            decoded.requires_grad = True

            if self.W_skip is not None:
                skip = self.compute_skip(input_acts)
                decoded = decoded + skip

            return decoded
        else:
            transcoder_acts, indices, _ = self.encode(input_acts)
            decoded = self.decode(transcoder_acts, indices)
            decoded = decoded.detach()
            decoded.requires_grad = True
            return decoded
            


def load_gemma_scope_transcoder(
    path: str,
    layer: int,
    device: Optional[torch.device] = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
    revision: Optional[str] = None,
) -> SingleLayerTranscoder:
    if os.path.isfile(path):
        path_to_params = path
    else:
        path_to_params = hf_hub_download(
            repo_id="google/gemma-scope-2b-pt-transcoders",
            filename=path,
            revision=revision,
            force_download=False,
        )

    # load the parameters, have to rename the threshold key,
    # as ours is nested inside the activation_function module
    param_dict = np.load(path_to_params)
    param_dict = {k: torch.tensor(v, device=device, dtype=dtype) for k, v in param_dict.items()}
    param_dict["activation_function.threshold"] = param_dict["threshold"]
    del param_dict["threshold"]

    # create the transcoders
    d_model = param_dict["W_enc"].shape[0]
    d_transcoder = param_dict["W_enc"].shape[1]

    # dummy JumpReLU; will get loaded via load_state_dict
    activation_function = JumpReLU(0.0, 0.1)
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(d_model, d_transcoder, activation_function, layer)
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder


def load_relu_transcoder(
    path: str,
    layer: int,
    device: torch.device = torch.device("cuda"),
    dtype: Optional[torch.dtype] = torch.float32,
):
    k=192
    param_dict = load_file(path, device="cpu")
    

    param_dict["W_enc"] = param_dict.pop("encoder.weight")
    param_dict["b_enc"] = param_dict.pop("encoder.bias", None)
    param_dict["W_dec"] = param_dict["W_dec"] 
    param_dict["b_dec"] = param_dict.get("b_dec", None)

    param_dict["W_enc"] = param_dict["W_enc"].mT.contiguous()
    assert param_dict.get("log_thresholds") is None
    activation_function = F.relu
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            param_dict["W_enc"].shape[0],
            param_dict["W_enc"].shape[1],
            activation_function,
            layer,
            k,
            skip_connection="W_skip" in param_dict,
        )
    transcoder.load_state_dict(param_dict, assign=True)
    return transcoder.to(dtype).cpu()

TranscoderSettings = namedtuple(
    "TranscoderSettings", ["transcoders", "feature_input_hook", "feature_output_hook", "scan"]
)


def load_transcoder_set(
    transcoder_config_file: str,
    device: Optional[torch.device] = "cpu",
    dtype: Optional[torch.dtype] = torch.float32,
) -> TranscoderSettings:
    """Loads either a preset set of transformers, or a set specified by a file.

    Args:
        transcoder_config_file (str): _description_
        device (Optional[torch.device], optional): _description_. Defaults to torch.device('cuda').

    Returns:
        TranscoderSettings: A namedtuple consisting of the transcoder dict,
        and their feature input hook, feature output hook and associated scan.
    """

    scan = None
    # try to match a preset, and grab its config
    if transcoder_config_file == "gemma":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/gemmascope-l0-0.yaml"
        scan = "gemma-2-2b"
    elif transcoder_config_file == "llama3-8b":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/llama3_8B.yaml"
        scan = "llama-3-8b"
    elif transcoder_config_file == "llama":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/llama-relu.yaml"
        scan = "llama-3-131k-relu"
    elif transcoder_config_file == "llada1.5-8b":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/llada1.5_8B.yaml"
        scan = "llada1.5-8b"

    with open(transcoder_config_file, "r") as file:
        config = yaml.safe_load(file)

    sorted_transcoder_configs = sorted(config["transcoders"], key=lambda x: x["layer"])
    if scan is None:
        # the scan defaults to a list of transcoder ids, preceded by the model's name
        model_name_no_slash = config["model_name"].split("/")[-1]
        scan = [
            f"{model_name_no_slash}/{transcoder_config['id']}"
            for transcoder_config in sorted_transcoder_configs
        ]

    # hf_paths = [
    #     t["filepath"] for t in sorted_transcoder_configs if t["filepath"].startswith("hf://")
    # ]
    # local_map = download_hf_uris(hf_paths)

    transcoders = {}
    for transcoder_config in sorted_transcoder_configs:
        path = transcoder_config["filepath"]
        if path.startswith("hf://"):
            local_path = local_map[path]
            repo_id = parse_hf_uri(path).repo_id
            if "gemma-scope" in repo_id:
                transcoder = load_gemma_scope_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
            else:
                transcoder = load_relu_transcoder(
                    local_path, transcoder_config["layer"], device=device, dtype=dtype
                )
        else:
            transcoder = load_relu_transcoder(
                path, transcoder_config["layer"], device=device, dtype=dtype
            )
        assert transcoder.layer_idx not in transcoders, (
            f"Got multiple transcoders for layer {transcoder.layer_idx}"
        )
        transcoders[transcoder.layer_idx] = transcoder.cpu()

    # we don't know how many layers the model has, but we need all layers from 0 to max covered
    assert set(transcoders.keys()) == set(range(max(transcoders.keys()) + 1)), (
        f"Each layer should have a transcoder, but got transcoders for layers "
        f"{set(transcoders.keys())}"
    )
    feature_input_hook = config["feature_input_hook"]
    feature_output_hook = config["feature_output_hook"]
    return TranscoderSettings(transcoders, feature_input_hook, feature_output_hook, scan)
