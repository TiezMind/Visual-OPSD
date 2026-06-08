"""Visual-OPSD offline (cached-teacher) dataset.

Builds text-only student sequences from the interleaved parquet shards:
``input_image + question + full_text_thought + answer``.  Teacher
logprob loading is handled by the training script (which reads from a
pre-computed cache produced by
``scripts/visual_opsd/collect_traces.py``), not by the dataset itself.

This dataset only serves the *offline* / cached variant.  The
on-policy paper-default Visual-OPSD pipeline uses
``opsd_paired_dataset.OPSDPairedIterableDataset`` instead, since the
student completion is sampled on-the-fly and the teacher logprobs are
recomputed every step.
"""

import io
import os

from PIL import Image, ImageFile, PngImagePlugin

from .data_utils import pil_img2rgb
from .distributed_iterable_dataset import DistributedIterableDataset

Image.MAX_IMAGE_PIXELS = 200000000
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2 ** 20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte


class VisualOPSDOfflineIterableDataset(DistributedIterableDataset):

    def __init__(
        self,
        dataset_name,
        transform,
        tokenizer,
        data_dir_list,
        num_used_data,
        local_rank=0,
        world_size=1,
        num_workers=8,
        data_status=None,
        teacher_cache_dir=None,
        shuffle_lines=False,
        shuffle_seed=0,
        **kwargs,
    ):
        super().__init__(dataset_name, local_rank, world_size, num_workers)
        self.transform = transform
        self.tokenizer = tokenizer
        self.data_status = data_status

        self.data_paths = self._load_parquet_paths(data_dir_list, num_used_data)
        self.set_epoch()

    def _load_parquet_paths(self, data_dir_list, num_used_data):
        all_paths = []
        for data_dir, num_data in zip(data_dir_list, num_used_data):
            if not os.path.exists(data_dir):
                print(f"[Visual-OPSD] Warning: {data_dir} does not exist, skipping")
                continue
            import pyarrow.parquet as pq
            parquet_files = sorted([
                os.path.join(data_dir, f)
                for f in os.listdir(data_dir)
                if f.endswith(".parquet") and "train" in f
            ])
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

    def __iter__(self):
        import pyarrow.parquet as pq

        data_paths_per_worker, worker_id = self.get_data_paths_per_worker()
        if self.data_status is not None:
            row_start_id = self.data_status[worker_id] + 1
        else:
            row_start_id = 0

        transform_stride = self.transform.stride
        file_cache = {}

        print(
            f"rank-{self.local_rank} worker-{worker_id} dataset-{self.dataset_name}: "
            f"resuming at row#{row_start_id}, total={len(data_paths_per_worker)}"
        )

        while True:
            for row_idx, (pf_path, pf_row) in enumerate(
                data_paths_per_worker[row_start_id:], start=row_start_id
            ):
                try:
                    if pf_path not in file_cache:
                        file_cache[pf_path] = pq.read_table(pf_path)
                    table = file_cache[pf_path]
                    row = {col: table.column(col)[pf_row].as_py()
                           for col in table.column_names}
                except Exception:
                    continue

                pid = str(row.get("pid", f"{row_idx}"))
                question = row.get("question", "")
                answer = row.get("answer", "")
                full_text = row.get("full_text_only_thought", "")
                if not full_text or not answer:
                    continue

                try:
                    problem_image = Image.open(
                        io.BytesIO(row["problem_image_0"]["bytes"])
                    ).convert("RGB")
                except Exception:
                    continue

                image_tensor_list = []
                text_ids_list = []
                sequence_plan = []
                num_tokens = 0

                # Input image (ViT encoding)
                image_tensor = self.transform(pil_img2rgb(problem_image))
                image_tensor_list.append(image_tensor)
                h, w = image_tensor.shape[1:]
                num_tokens += w * h // transform_stride ** 2
                sequence_plan.append({
                    "type": "vit_image", "enable_cfg": 0, "loss": 0,
                    "special_token_loss": 0, "special_token_label": None,
                })

                # Question (no loss)
                q_ids = self.tokenizer.encode(question)
                if q_ids:
                    text_ids_list.append(q_ids)
                    num_tokens += len(q_ids)
                    sequence_plan.append({
                        "type": "text", "enable_cfg": 0, "loss": 0,
                        "special_token_loss": 0, "special_token_label": None,
                    })

                # Full text thought (with loss)
                t_ids = self.tokenizer.encode(full_text)
                if t_ids:
                    text_ids_list.append(t_ids)
                    num_tokens += len(t_ids)
                    sequence_plan.append({
                        "type": "text", "enable_cfg": 0, "loss": 1,
                        "special_token_loss": 0, "special_token_label": None,
                    })

                # Answer (with loss)
                a_ids = self.tokenizer.encode(answer)
                if a_ids:
                    text_ids_list.append(a_ids)
                    num_tokens += len(a_ids)
                    sequence_plan.append({
                        "type": "text", "enable_cfg": 0, "loss": 1,
                        "special_token_loss": 0, "special_token_label": None,
                    })

                yield dict(
                    image_tensor_list=image_tensor_list,
                    text_ids_list=text_ids_list,
                    sequence_plan=sequence_plan,
                    num_tokens=num_tokens,
                    data_indexes={
                        "data_indexes": row_idx,
                        "worker_id": worker_id,
                        "dataset_name": self.dataset_name,
                        "pid": pid,
                    },
                )

            row_start_id = 0
            file_cache = {}
            print(f"{self.dataset_name} repeat in rank-{self.local_rank} worker-{worker_id}")
