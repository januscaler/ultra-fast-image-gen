"""Benchmark one uncensored-TE quant in its own process (so peak RSS is isolated).

Usage: python bench_uncensored.py <q4_k_m|q6_k|q8_0> [steps] [seed]
Prints a line starting with "METRICS " followed by JSON.
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from loaders import (
    UNCENSORED_TE_QUANTS,
    UNCENSORED_TE_REPO,
    load_flux2_klein_uncensored_pipeline,
)

QUANT = sys.argv[1]
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 28
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 1234
PROMPT = "a photograph of a mountain landscape at sunset, highly detailed"


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9


def mps_alloc_gb():
    try:
        return torch.mps.current_allocated_memory() / 1e9
    except Exception:
        return 0.0


def gguf_size_gb():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(UNCENSORED_TE_REPO, UNCENSORED_TE_QUANTS[QUANT])
    return os.path.getsize(p) / 1e9


t0 = time.time()
pipe = load_flux2_klein_uncensored_pipeline(device="mps", quant=QUANT)
load_s = time.time() - t0
rss_after_load = rss_gb()
mps_after_load = mps_alloc_gb()

# Isolate the text-encoder cost: this is the part that actually varies with the
# quant (the diffusion backbone is identical across runs). Time a couple of
# forwards after a warm-up.
te_encode_s = None
try:
    ids = pipe.tokenizer(PROMPT, return_tensors="pt").to("mps")
    with torch.inference_mode():
        pipe.text_encoder(**ids, output_hidden_states=True)  # warm-up
        torch.mps.synchronize()
        t0 = time.time()
        for _ in range(3):
            pipe.text_encoder(**ids, output_hidden_states=True)
        torch.mps.synchronize()
        te_encode_s = round((time.time() - t0) / 3, 3)
except Exception as e:
    print("te-encode timing skipped:", e)

gen = torch.Generator("mps").manual_seed(SEED)
t0 = time.time()
with torch.inference_mode():
    img = pipe(
        prompt=PROMPT,
        height=512,
        width=512,
        num_inference_steps=STEPS,
        guidance_scale=3.5,
        generator=gen,
    ).images[0]
gen_s = time.time() - t0

out = f"/tmp/uncensored_{QUANT}.png"
img.save(out)

metrics = {
    "quant": QUANT,
    "steps": STEPS,
    "gguf_disk_gb": round(gguf_size_gb(), 2),
    "load_s": round(load_s, 1),
    "te_encode_s": te_encode_s,
    "gen_s": round(gen_s, 1),
    "mps_alloc_after_load_gb": round(mps_after_load, 2),
    "mps_alloc_peak_gb": round(mps_alloc_gb(), 2),
    "peak_rss_gb": round(rss_gb(), 2),
    "out": out,
}
print("METRICS " + json.dumps(metrics))
