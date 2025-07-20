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

import circuit_tracer
from circuit_tracer.transcoder.activation_functions import JumpReLU
from circuit_tracer.utils.hf_utils import download_hf_uris, parse_hf_uri


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

        self.W_enc = nn.Parameter(torch.zeros(d_model, d_transcoder))
        self.W_dec = nn.Parameter(torch.zeros(d_transcoder, d_model))
        self.b_enc = nn.Parameter(torch.zeros(d_transcoder))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

        if skip_connection:
            self.W_skip = nn.Parameter(torch.zeros(d_model, d_model))
        else:
            self.W_skip = None

        self.activation_function = activation_function

    def encode(self, input_acts, k=0, apply_activation_function: bool = True):
        
        pre_acts = input_acts.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        if not apply_activation_function:
            return pre_acts
        acts = self.activation_function(pre_acts)
        if k != 0:
            acts, indices = torch.topk(acts, k, dim=-1, sorted=False)
            return acts, indices
        else:
            return acts

    def decode(self, acts, indices=None):
        def eager_decode(top_indices, top_acts, W_dec):
            return nn.functional.embedding_bag(
                top_indices, W_dec.mT, per_sample_weights=top_acts, mode="sum"
            )
        if indices is not None:
            y = eager_decode(indices, acts.to(self.dtype), self.W_dec.mT)
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

    def forward(self, input_acts, k=0):
        if k == 0:
            transcoder_acts = self.encode(input_acts)
            decoded = self.decode(transcoder_acts)
            decoded = decoded.detach()
            decoded.requires_grad = True

            if self.W_skip is not None:
                skip = self.compute_skip(input_acts)
                decoded = decoded + skip

            return decoded
        else:
            transcoder_acts, indices = self.encode(input_acts, k)
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
    param_dict = load_file(path, device="cpu")
    

    param_dict["W_enc"] = param_dict.pop("encoder.weight")
    param_dict["b_enc"] = param_dict.pop("encoder.bias", None)
    param_dict["W_dec"] = param_dict["W_dec"]  # 已经存在就不动
    param_dict["b_dec"] = param_dict.get("b_dec", None)

    param_dict["W_enc"] = param_dict["W_enc"].T.contiguous()
    param_dict["W_dec"] = param_dict["W_dec"].T.contiguous()

    assert param_dict.get("log_thresholds") is None
    activation_function = F.relu
    with torch.device("meta"):
        transcoder = SingleLayerTranscoder(
            param_dict["W_enc"].shape[0],
            param_dict["W_enc"].shape[1],
            activation_function,
            layer,
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
    elif transcoder_config_file == "llada":
        package_path = resources.files(circuit_tracer)
        transcoder_config_file = package_path / "configs/llada1.5_8B.yaml"

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
