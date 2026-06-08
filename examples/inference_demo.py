"""Minimal text-only inference demo for a Visual-OPSD-trained student.

After Visual-OPSD training, the student is a pure understanding model —
it ingests a problem image plus a question and emits a text reasoning
trace ending in ``<answer>...</answer>``. The VAE / diffusion pathway
is never used, so inference is fast (10s / sample on a single H800)
and memory-friendly (~30 GB peak).

Usage
-----
    python examples/inference_demo.py \
        --model_path results/visual-opsd-ema-beta0.5/checkpoints/0001000 \
        --image path/to/problem.jpg \
        --question "Which of the two objects is taller, A or B?"

For full interleaved inference (text + generated VT) with the *base*
ThinkMorph teacher, use ``inferencer.py`` directly — it carries the
original BAGEL API and runs the diffusion pathway. For batch evaluation
across the 9 benchmarks reported in the paper, use the
VLMEvalKit-ThinkMorph harness (see ``EVAL.md``).
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from safetensors.torch import load_file

# Make the repo importable when this script is invoked from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_utils import add_special_tokens, pil_img2rgb
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
from inferencer import InterleaveInferencer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_path", default="models/ThinkMorph-7B",
                   help="Path to either the base ThinkMorph-7B checkpoint "
                        "or a Visual-OPSD student checkpoint directory.")
    p.add_argument("--image", required=True, help="Path to the input image.")
    p.add_argument("--question", required=True, help="Question text.")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--do_sample", action="store_true")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


def build_understanding_model(model_path: str, device: str) -> tuple:
    """Load the Bagel UMM in understanding-only mode (no VAE)."""
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.freeze_und = False
    language_model = Qwen2ForCausalLM(llm_config)

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + (-2)
    vit_config.rope = False
    vit_model = SiglipVisionModel(vit_config)

    config = BagelConfig(
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
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, num_new = add_special_tokens(tokenizer)
    if num_new > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # The base checkpoint may ship a single ``model.safetensors`` or a
    # sharded layout; handle both.
    single = os.path.join(model_path, "model.safetensors")
    if os.path.isfile(single):
        sd = load_file(single, device="cpu")
    else:
        sd = {}
        for fname in sorted(os.listdir(model_path)):
            if fname.endswith(".safetensors") and fname != "ae.safetensors":
                sd.update(load_file(os.path.join(model_path, fname), device="cpu"))
    sd.pop("latent_pos_embed.pos_embed", None)
    sd.pop("vit_pos_embed.pos_embed", None)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load_state_dict] missing={len(missing)}, unexpected={len(unexpected)}")

    model = model.to(device, dtype=torch.bfloat16).eval()
    vit_transform = ImageTransform(stride=14, max_image_size=980, min_image_size=378)
    return model, tokenizer, new_token_ids, vit_transform


def main() -> None:
    args = parse_args()

    if not os.path.isdir(args.model_path):
        sys.exit(f"Model path not found: {args.model_path}")
    if not os.path.isfile(args.image):
        sys.exit(f"Image not found: {args.image}")

    model, tokenizer, new_token_ids, vit_transform = build_understanding_model(
        args.model_path, args.device
    )

    # Visual-OPSD students do not use the VAE; we pass ``vae_transform=vit_transform``
    # purely to satisfy the InterleaveInferencer constructor signature.
    # ``understanding_output=True`` skips the diffusion pathway.
    inferencer = InterleaveInferencer(
        model=model,
        vae_model=None,
        tokenizer=tokenizer,
        vae_transform=vit_transform,
        vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )

    image = Image.open(args.image)
    image = pil_img2rgb(image)
    output_list = inferencer.interleave_inference(
        input_lists=[image, args.question],
        think=True,
        understanding_output=True,
        max_think_token_n=args.max_new_tokens,
        do_sample=args.do_sample,
        text_temperature=args.temperature,
    )

    print("\n=== Model output ===")
    for piece in output_list:
        if isinstance(piece, str):
            print(piece)
        else:
            print(f"[image generated: {getattr(piece, 'size', '?')}]")


if __name__ == "__main__":
    main()
