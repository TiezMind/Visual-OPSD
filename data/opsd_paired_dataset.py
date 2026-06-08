"""Visual-OPSD Paired Dataset (on-policy version).

This dataset emits RAW per-sample envelopes.  Unlike the first-iteration
version (which pre-packed student/teacher sequences with ground-truth
completions), the training loop now:

  1. Samples a completion from the CURRENT student weights.
  2. Packs the student/teacher full sequences with THAT completion.
  3. Forwards both, computes JSD over the entire completion span.

So we no longer pack inside the dataset; we just hand back the raw bits
the builder needs.  The envelope shape is:

    {
      "problem_image_tensor":     torch.Tensor [3, H, W]  (ViT-transformed),
      "question_text":            str,
      "question_ids":             List[int]  (tokenized question — no bos/eos),
      "reference_thoughts_ids":   List[List[int]]  (tokenized reasoning thoughts),
      "reference_vt_tensors":     List[torch.Tensor | None]  (ViT-transformed VT imgs),
      "reference_answer_ids":     List[int]  (tokenized ground-truth answer — for the
                                              privileged teacher context),
      "reference_answer_text":    str,  (kept for logging),
      "data_indexes":             {...},
    }

The text ids are produced via ``tokenizer.encode(text, add_special_tokens=False)``
— BAGEL adds BOS/EOS inside the pack builder / inference primitives, so we
hand over clean subword ids here.
"""

from __future__ import annotations

import io
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset
from .transforms import ImageTransform

Image.MAX_IMAGE_PIXELS = 200_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class OPSDPairedIterableDataset(DistributedIterableDataset):
    """Yields RAW per-sample envelopes (no packing, no completion attached)."""

    def __init__(
        self,
        dataset_name: str,
        transform,
        tokenizer,
        data_dir_list: List[str],
        num_used_data: List[int],
        local_rank: int = 0,
        world_size: int = 1,
        num_workers: int = 8,
        data_status: Optional[Dict[int, int]] = None,
        max_reference_rounds: int = 8,
        **kwargs,
    ) -> None:
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status
        self.max_reference_rounds = max_reference_rounds
        self.data_paths = self._load_parquet_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def _load_parquet_paths(
        self, data_dir_list: List[str], num_used_data: List[int]
    ) -> List[Tuple[str, int]]:
        """Collect (parquet_path, row_idx) pairs up to each dir's per-dir budget."""
        import pyarrow.parquet as pq

        all_paths: List[Tuple[str, int]] = []
        for data_dir, num_data in zip(data_dir_list, num_used_data):
            if not os.path.exists(data_dir):
                print(f"[OPSDPaired] Warning: {data_dir} missing, skipping")
                continue
            parquet_files = sorted(
                [
                    os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".parquet") and "train" in f
                ]
            )
            count = 0
            for pf in parquet_files:
                table = pq.read_table(pf)
                for i in range(len(table)):
                    if count >= num_data:
                        break
                    all_paths.append((pf, i))
                    count += 1
                if count >= num_data:
                    break
        return all_paths

    def _encode(self, text: str) -> List[int]:
        """Tokenize without adding bos/eos; those are inserted during packing."""
        if not text:
            return []
        return self.tokenizer.encode(text, add_special_tokens=False)

    def __iter__(self):
        import pyarrow.parquet as pq

        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        row_start_id = (
            self.data_status[worker_id] + 1 if self.data_status is not None else 0
        )

        file_cache: Dict[str, Any] = {}
        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resume row#{row_start_id}, total={len(data_paths_per_worker)}"
        )

        while True:
            for row_idx, (pf_path, pf_row) in enumerate(
                data_paths_per_worker[row_start_id:], start=row_start_id
            ):
                try:
                    if pf_path not in file_cache:
                        file_cache[pf_path] = pq.read_table(pf_path)
                    table = file_cache[pf_path]
                    row = {
                        col: table.column(col)[pf_row].as_py()
                        for col in table.column_names
                    }
                except Exception:
                    continue

                pid = str(row.get("pid", f"{row_idx}"))
                question = row.get("question", "") or ""
                answer = row.get("answer", "") or ""
                if not answer or not question:
                    continue

                try:
                    problem_image = Image.open(
                        io.BytesIO(row["problem_image_0"]["bytes"])
                    ).convert("RGB")
                    image_tensor = self.transform(pil_img2rgb(problem_image))
                except Exception:
                    continue

                thoughts_texts: List[str] = []
                vt_image_tensors: List = []
                for i in range(self.max_reference_rounds):
                    tkey = f"resoning_thought_{i}"
                    ikey = f"reasoning_image_{i}"
                    thought = row.get(tkey) or ""
                    if thought:
                        thoughts_texts.append(thought)
                    img_bytes = row.get(ikey)
                    if img_bytes:
                        try:
                            img = Image.open(
                                io.BytesIO(img_bytes["bytes"])
                            ).convert("RGB")
                            vt_image_tensors.append(
                                self.transform(pil_img2rgb(img))
                            )
                        except Exception:
                            vt_image_tensors.append(None)

                if not thoughts_texts:
                    continue

                question_ids = self._encode(question)
                reference_thoughts_ids = [self._encode(t) for t in thoughts_texts]
                reference_answer_ids = self._encode(answer)
                if not question_ids or not reference_answer_ids:
                    continue

                yield dict(
                    problem_image_tensor=image_tensor,
                    question_text=question,
                    question_ids=question_ids,
                    reference_thoughts_ids=reference_thoughts_ids,
                    reference_thoughts_texts=thoughts_texts,
                    reference_vt_tensors=vt_image_tensors,
                    reference_answer_ids=reference_answer_ids,
                    reference_answer_text=answer,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                        "pid": pid,
                    },
                )

            row_start_id = 0
            file_cache = {}
            print(
                f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}"
            )


# --------------------------------------------------------------------- #
# Helper: multi-group sampler
# --------------------------------------------------------------------- #


class OPSDMultiGroupDataset(torch.utils.data.IterableDataset):
    """Wrap a list of :class:`OPSDPairedIterableDataset` instances with
    weighted group sampling.

    Mirrors the group-weighting logic of ``PackedDataset`` but emits RAW
    per-sample envelopes (no packing).  Each call to ``__iter__`` returns
    an iterator that selects a group according to ``grouped_weights`` and
    forwards the next raw sample from that group.
    """

    def __init__(
        self,
        grouped_datasets: List["OPSDPairedIterableDataset"],
        grouped_weights: Optional[List[float]] = None,
        data_seed: int = 42,
    ) -> None:
        super().__init__()
        assert len(grouped_datasets) > 0, "need at least one group"
        self.grouped_datasets = grouped_datasets
        if grouped_weights is None:
            grouped_weights = [1.0 / len(grouped_datasets)] * len(grouped_datasets)
        total = sum(grouped_weights)
        assert total > 0.0
        self.grouped_weights = [w / total for w in grouped_weights]
        self._cumprobs = [
            sum(self.grouped_weights[: i + 1]) for i in range(len(self.grouped_weights))
        ]
        self._data_seed = data_seed

    def set_epoch(self, seed: int) -> None:
        self._data_seed = seed
        for ds in self.grouped_datasets:
            ds.set_epoch(seed)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        iters = [iter(ds) for ds in self.grouped_datasets]
        rng = random.Random(self._data_seed)
        while True:
            r = rng.random()
            idx = 0
            for i, c in enumerate(self._cumprobs):
                if r < c:
                    idx = i
                    break
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.grouped_datasets[idx])
                yield next(iters[idx])


def build_opsd_raw_dataset(
    dataset_meta: Dict[str, Dict[str, Any]],
    tokenizer,
    local_rank: int = 0,
    world_size: int = 1,
    num_workers: int = 4,
    data_status: Optional[Dict[str, Any]] = None,
) -> OPSDMultiGroupDataset:
    """Build a raw (no-pack) multi-group dataset from a YAML ``dataset_meta``.

    ``dataset_meta`` is the direct ``yaml.safe_load`` result — a mapping of
    ``grouped_dataset_name -> kwargs`` as used by ``PackedDataset``.  This
    helper replicates the DATASET_REGISTRY dispatch but skips the buffer /
    pack loop so we get raw envelopes to feed to the on-policy sampler.
    """
    from .dataset_info import DATASET_INFO, DATASET_REGISTRY

    import copy as _copy
    meta = _copy.deepcopy(dataset_meta)

    grouped_datasets: List[OPSDPairedIterableDataset] = []
    grouped_weights: List[float] = []
    for grouped_name, dataset_args in meta.items():
        dataset_args.pop("is_mandatory", None)
        weight = float(dataset_args.pop("weight", 1.0))
        grouped_weights.append(weight)

        if "image_transform_args" in dataset_args:
            transform = ImageTransform(**dataset_args.pop("image_transform_args"))
            dataset_args["transform"] = transform

        assert "dataset_names" in dataset_args, (
            f"{grouped_name}: dataset_names missing from config"
        )
        dataset_names = dataset_args.pop("dataset_names")
        data_dir_list: List[str] = []
        for name in dataset_names:
            info = DATASET_INFO[grouped_name][name]
            data_dir_list.append(info["data_dir"])
        dataset_args["data_dir_list"] = data_dir_list

        resume = dataset_args.pop("resume_data_status", True)
        status_for_group = None
        if data_status is not None and grouped_name in data_status and resume:
            status_for_group = data_status[grouped_name]

        cls = DATASET_REGISTRY[grouped_name]
        dataset = cls(
            dataset_name=grouped_name,
            tokenizer=tokenizer,
            local_rank=local_rank,
            world_size=world_size,
            num_workers=num_workers,
            data_status=status_for_group,
            **dataset_args,
        )
        grouped_datasets.append(dataset)

    return OPSDMultiGroupDataset(
        grouped_datasets=grouped_datasets,
        grouped_weights=grouped_weights,
    )
