"""Download the base ThinkMorph-7B checkpoint from Hugging Face.

This is the *unmodified* unified multimodal model that Visual-OPSD uses
as both the teacher (with privileged VT context) and the starting point
of the student. After Visual-OPSD training, your distilled student
checkpoint lives under ``results/<run_name>/checkpoints/``.

Usage
-----
    python download_model.py                              # → models/ThinkMorph-7B/
    python download_model.py --save-dir /elsewhere/TM-7B  # custom location
"""

import argparse
import os

from huggingface_hub import snapshot_download


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download the ThinkMorph-7B checkpoint used as the "
        "base UMM for Visual-OPSD training and inference."
    )
    p.add_argument("--repo-id", type=str, default="ThinkMorph/ThinkMorph-7B")
    p.add_argument("--save-dir", type=str, default="models/ThinkMorph-7B")
    p.add_argument(
        "--allow-pattern",
        action="append",
        default=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
        help="May be passed multiple times.",
    )
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", action="store_false", dest="resume")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    save_dir = args.save_dir
    repo_id = args.repo_id
    cache_dir = os.path.join(save_dir, "cache")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Downloading {repo_id} → {save_dir}")
    snapshot_download(
        cache_dir=cache_dir,
        local_dir=save_dir,
        repo_id=repo_id,
        local_dir_use_symlinks=False,
        resume_download=args.resume,
        allow_patterns=args.allow_pattern,
    )
    print(
        f"Done. Pass --model_path {save_dir} to any Visual-OPSD training "
        f"or inference script."
    )


if __name__ == "__main__":
    main()
