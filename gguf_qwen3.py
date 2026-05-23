"""
Load an abliterated Qwen3 text encoder from a GGUF file with the weights kept
QUANTIZED in memory and dequantized on the fly during the forward pass (the same
trick ComfyUI-GGUF uses). This is what makes RAM and speed actually scale with
the chosen quant (q4_k_m / q6_k / q8_0) instead of all dequantizing to bf16 on
load the way transformers' built-in GGUF loader does.

The dequant kernels are reused from ComfyUI-GGUF (Apache-2.0); see
vendor/gguf_dequant.py. The ggml->transformers tensor-name mapping is reused
from transformers itself.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gguf
from accelerate import init_empty_weights
from transformers import AutoConfig, Qwen3ForCausalLM
from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vendor"))
from gguf_dequant import dequantize, dequantize_functions  # noqa: E402

_FLOAT_QTYPES = (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16)


def _raw_bytes(t) -> torch.Tensor:
    """Raw quantized block bytes of a gguf tensor as a 1-D uint8 torch tensor."""
    return torch.from_numpy(np.ascontiguousarray(t.data)).view(torch.uint8).clone()


def _full_value(t, dtype) -> torch.Tensor:
    """Materialize a gguf tensor to a real tensor (dequantizing if needed).

    Used for the things we keep dense: norms (F32) and the token embedding.
    Shape is taken from gguf's ne order reversed -> torch (rows, cols).
    """
    oshape = tuple(int(x) for x in reversed(t.shape))
    if t.tensor_type in _FLOAT_QTYPES:
        return torch.from_numpy(np.ascontiguousarray(t.data)).to(dtype).reshape(oshape)
    return dequantize(_raw_bytes(t), t.tensor_type, oshape, dtype)


class GGMLLinear(nn.Module):
    """nn.Linear replacement holding a quantized GGUF weight, dequantized per call."""

    def __init__(self, qbytes, qtype, out_features, in_features, compute_dtype):
        super().__init__()
        # persistent=False: this buffer is loaded from the GGUF, not a checkpoint.
        self.register_buffer("qweight", qbytes, persistent=False)
        self.qtype = qtype
        self.out_features = out_features
        self.in_features = in_features
        self.compute_dtype = compute_dtype

    def forward(self, x):
        w = dequantize(
            self.qweight, self.qtype, (self.out_features, self.in_features), self.compute_dtype
        )
        return F.linear(x, w.to(x.dtype))

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, qtype={self.qtype.name}"


def _set_submodule(root, path, new):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    setattr(parent, parts[-1], new)


def _assign_param(root, path, value):
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    parent.register_parameter(parts[-1], nn.Parameter(value, requires_grad=False))


def load_qwen3_gguf_text_encoder(gguf_path, config_dir, device="mps", compute_dtype=torch.bfloat16):
    """Build a Qwen3ForCausalLM whose Linear weights stay quantized in RAM.

    gguf_path:  path to the .gguf text-encoder file.
    config_dir: dir with the Qwen3 config.json (the repo's safetensors subfolder).
    """
    cfg = AutoConfig.from_pretrained(config_dir)
    with init_empty_weights():
        model = Qwen3ForCausalLM(cfg)

    name_map = get_gguf_hf_weights_map(model, "qwen3", cfg.num_hidden_layers)
    hf_to_ggml = {hf: ggml for ggml, hf in name_map.items()}

    reader = gguf.GGUFReader(gguf_path)
    tensors = {t.name: t for t in reader.tensors}

    quant_counts = {}

    # 1) Swap every Linear for a GGMLLinear that holds the quantized weight.
    for path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        ggml_name = hf_to_ggml.get(path + ".weight")
        t = tensors.get(ggml_name)
        if t is None:  # e.g. tied lm_head with no own tensor
            continue
        qbytes = _raw_bytes(t).to(device)
        _set_submodule(
            model,
            path,
            GGMLLinear(qbytes, t.tensor_type, module.out_features, module.in_features, compute_dtype),
        )
        quant_counts[t.tensor_type.name] = quant_counts.get(t.tensor_type.name, 0) + 1

    # 2) Materialize the dense bits (token embedding + all norms) on device.
    for pname, p in list(model.named_parameters()):
        if p.device.type != "meta":
            continue
        ggml_name = hf_to_ggml.get(pname)
        t = tensors.get(ggml_name)
        if t is None:  # tied weights handled by tie_weights() below
            continue
        _assign_param(model, pname, _full_value(t, compute_dtype).to(device))

    model.tie_weights()
    model.eval()
    model.to(device)
    return model, quant_counts
