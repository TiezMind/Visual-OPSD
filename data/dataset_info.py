# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Dataset registry for Visual-OPSD training.

Each training dataset is registered in two places:

  * ``DATASET_REGISTRY`` — maps the YAML group name (e.g.
    ``visual_opsd`` / ``visual_opsd_offline``) to the IterableDataset
    class that knows how to read that group's parquet files.
  * ``DATASET_INFO``     — maps ``(group_name, sub_dataset_name)`` to
    a small dict with the on-disk paths and sample counts.

Before you start training, edit the ``data_dir`` / ``jsonl_path`` fields
below so that they point to where you downloaded each dataset.  The
default paths use ``./datasets/<DatasetName>/...`` relative to the repo
root, which is what the documented download instructions produce.
"""

import os

from .interleave_datasets import UnifiedEditIterableDataset
from .opsd_paired_dataset import OPSDPairedIterableDataset
from .t2i_dataset import T2IIterableDataset
from .visual_opsd_offline_dataset import VisualOPSDOfflineIterableDataset
from .vlm_dataset import SftJSONLIterableDataset


DATASET_REGISTRY = {
    "t2i_pretrain": T2IIterableDataset,
    "vlm_sft": SftJSONLIterableDataset,
    "unified_edit": UnifiedEditIterableDataset,
    "visual_opsd_offline": VisualOPSDOfflineIterableDataset,
    "visual_opsd": OPSDPairedIterableDataset,
}


# ---------------------------------------------------------------------- #
# Data root resolution
# ---------------------------------------------------------------------- #
#
# By default, datasets are expected under ``./datasets/<DatasetName>/`` at
# the repo root (matching the layout in ``docs/DATA.md``).  You can override
# that by setting the ``VISUAL_OPSD_DATA_ROOT`` environment variable, e.g.
#
#     export VISUAL_OPSD_DATA_ROOT=/path/to/your/datasets
#
DATA_ROOT = os.environ.get("VISUAL_OPSD_DATA_ROOT", "datasets")


def _parquet(dataset_name: str) -> str:
    return os.path.join(DATA_ROOT, dataset_name, "data")


def _images(dataset_name: str) -> str:
    return os.path.join(DATA_ROOT, dataset_name, "images")


def _jsonl(dataset_name: str) -> str:
    return os.path.join(DATA_ROOT, dataset_name, f"{dataset_name}.jsonl")


# Four reasoning datasets used in the Visual-OPSD paper.  Sample counts
# match the ThinkMorph release on Hugging Face.
_THINKMORPH_DATASETS = {
    "Visual_Search":      6990,
    "Spatial_Navigation": 6000,
    "Jigsaw_Assembly":    6000,
    "Chart_Refocus":      6000,
}


DATASET_INFO = {
    # ------------------------------------------------------------------ #
    # unified_edit / vlm_sft are inherited from the BAGEL training stack.
    # They are not used by the Visual-OPSD pipeline directly; we keep
    # example entries here so ``DATASET_REGISTRY`` stays self-consistent.
    # ------------------------------------------------------------------ #
    "unified_edit": {
        # Example placeholder — replace with your own paths.
        "seedxedit_multi": {
            "data_dir": os.path.join(DATA_ROOT, "bagel_example/editing/seedxedit_multi"),
            "num_files": 10,
            "num_total_samples": 1000,
            "parquet_info_path": os.path.join(
                DATA_ROOT, "bagel_example/editing/parquet_info/seedxedit_multi.json"
            ),
        },
    },

    "vlm_sft": {
        name: {
            "data_dir": _images(name),
            "jsonl_path": _jsonl(name),
            "num_total_samples": n,
        }
        for name, n in _THINKMORPH_DATASETS.items()
    },

    # ------------------------------------------------------------------ #
    # Visual-OPSD (offline / cached-teacher variant): reads the same
    # parquet shards as ``visual_opsd`` but builds text-only student
    # sequences and looks up teacher logprobs from a pre-computed cache
    # (see ``scripts/visual_opsd/collect_traces.py``).
    # ------------------------------------------------------------------ #
    "visual_opsd_offline": {
        name: {
            "data_dir": _parquet(name),
            "num_total_samples": n,
        }
        for name, n in _THINKMORPH_DATASETS.items()
    },

    # ------------------------------------------------------------------ #
    # Visual-OPSD (on-policy, paper default): reads the same parquet and
    # emits raw envelopes (problem image + question + reference thoughts
    # + VT images + answer) which the training loop pairs with the
    # student's on-policy completion.
    # ------------------------------------------------------------------ #
    "visual_opsd": {
        name: {
            "data_dir": _parquet(name),
            "num_total_samples": n,
        }
        for name, n in _THINKMORPH_DATASETS.items()
    },
}
