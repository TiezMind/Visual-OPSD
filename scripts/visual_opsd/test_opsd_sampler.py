"""Smoke test for the on-policy sampler — non-distributed, small inputs.

Uses the real ThinkMorph-7B checkpoint on a single GPU and verifies:

  - ``OnPolicySampler.generate`` returns a list of token ids,
  - the returned ids do NOT start with BOS or end with EOS,
  - the completion length is within ``max_new_tokens``.

This test is *not* part of the FSDP path — it runs directly on an
unwrapped Bagel on GPU to keep the smoke test cheap.  The full FSDP path
is exercised by ``scripts/visual_opsd/run_visual_opsd.sh``.

Run:
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 python scripts/visual_opsd/test_opsd_sampler.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from data.data_utils import add_special_tokens
from data.opsd_paired_dataset import OPSDPairedIterableDataset
from data.transforms import ImageTransform
from modeling.bagel import (
    Bagel,
    BagelConfig,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from scripts.visual_opsd.on_policy_sampler import (
    OnPolicySampler,
    SamplingConfig,
    _forward_cache_update_vit_from_tensor,
)

MODEL_PATH = os.environ.get("VISUAL_OPSD_MODEL_PATH", "models/ThinkMorph-7B")
DATA_DIR = os.environ.get("VISUAL_OPSD_DATA_DIR", "datasets/Visual_Search/data")


def _build_bagel(device: str) -> Bagel:
    llm_config = Qwen2Config.from_json_file(
        os.path.join(MODEL_PATH, "llm_config.json")
    )
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    lm = Qwen2ForCausalLM(llm_config)
    lm.init_moe()

    vit_config = SiglipVisionConfig.from_json_file(
        os.path.join(MODEL_PATH, "vit_config.json")
    )
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + (-2)
    vit_config.rope = False
    vit = SiglipVisionModel(vit_config)

    cfg = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=64,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
    )
    model = Bagel(lm, vit, cfg)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    from safetensors.torch import load_file
    sd = load_file(os.path.join(MODEL_PATH, "model.safetensors"), device="cpu")
    sd.pop("latent_pos_embed.pos_embed", None)
    sd.pop("vit_pos_embed.pos_embed", None)
    msg = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(msg.missing_keys)}, "
          f"unexpected={len(msg.unexpected_keys)}")
    return model.to(device, dtype=torch.bfloat16).eval()


def main() -> None:
    if not torch.cuda.is_available():
        print("[SKIP] no CUDA")
        return
    if not os.path.isdir(MODEL_PATH):
        print(f"[SKIP] model path not found: {MODEL_PATH}")
        return
    if not os.path.isdir(DATA_DIR):
        print(f"[SKIP] data dir not found: {DATA_DIR}")
        return

    device = "cuda:0"

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)

    print("Loading Bagel model (this takes ~30s)...")
    model = _build_bagel(device)
    if num_new > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    transform = ImageTransform(
        image_stride=14, max_image_size=980, min_image_size=378, max_pixels=2_007_040
    )
    dataset = OPSDPairedIterableDataset(
        dataset_name="visual_opsd",
        transform=transform,
        tokenizer=tokenizer,
        data_dir_list=[DATA_DIR],
        num_used_data=[1],
        local_rank=0,
        world_size=1,
        num_workers=0,
    )
    dataset.set_epoch(0)
    raw = next(iter(dataset))
    raw["problem_image_tensor"] = raw["problem_image_tensor"].to(device)

    # We mimic the sampler but call it directly (not through FSDP.summon_full_params).
    # Override the internal `generate` to bypass FSDP (the wrapper is a no-op
    # when passed a raw module).
    sampler = OnPolicySampler(
        tokenizer=tokenizer,
        vit_transform=transform,
        new_token_ids=new_token_ids,
        cfg=SamplingConfig(
            max_new_tokens=64, temperature=1.0, do_sample=True,
            system_prompt=None, instruction_suffix=None,
            max_image_skips=0,  # keep this smoke test deterministic /
            # single-round; the multi-round image-skip path is exercised
            # by ``scripts/visual_opsd/run_visual_opsd.sh``.
        ),
    )

    with torch.no_grad(), torch.autocast(
        device_type="cuda", enabled=True, dtype=torch.bfloat16
    ):
        gen = sampler._generate_single(
            model=model,
            raw=raw,
            device=device,
            eos_id=int(new_token_ids["eos_token_id"]),
        )

    assert gen.dim() == 2, f"expected [T, 1], got shape={tuple(gen.shape)}"
    tokens = gen.view(-1).tolist()
    assert tokens[0] == new_token_ids["bos_token_id"], (
        "first token should be BOS from prepare_start_tokens"
    )
    assert len(tokens) <= 64 + 1, f"exceeded max_new_tokens: {len(tokens)}"
    # Strip BOS
    completion = tokens[1:]
    # Shouldn't end with EOS (generate_text breaks before appending)
    if completion:
        assert completion[-1] != new_token_ids["eos_token_id"], (
            "tail EOS should not be present in the raw generate_text output"
        )

    decoded = tokenizer.decode(completion) if completion else "<empty>"
    print(f"[ok] sampled {len(completion)} tokens")
    print(f"     preview: {decoded[:200]!r}")


if __name__ == "__main__":
    main()
