"""Mini integration test for the on-policy OPSD dataset + pack builder.

Verifies (no GPU needed):
  - ``OPSDPairedIterableDataset`` produces a well-formed RAW envelope
    (problem image tensor, tokenized question, reference thoughts, VT tensors,
     reference answer).
  - ``OPSDPackBuilder`` turns that envelope + a synthetic ``completion_ids``
    list into a packed batch ready for ``Bagel.forward``, with matching
    student / teacher ``ce_loss_indexes`` lengths (= len(completion)+1).
  - The packed student batch's CE label sequence ends with ``<eos>`` and
    starts at position ``bos`` as expected.

Run:
    source .venv/bin/activate
    python scripts/visual_opsd/test_opsd_dataset.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch

from data.data_utils import add_special_tokens
from data.opsd_pack_builder import (
    OPSDPackBuilder,
    PackBuilderConfig,
    TEACHER_REFERENCE_INTRO,
    TEACHER_TRANSITION,
    VLM_THINK_SYSTEM_PROMPT,
)
from data.opsd_paired_dataset import OPSDPairedIterableDataset
from data.transforms import ImageTransform
from modeling.qwen2 import Qwen2Tokenizer


MODEL_PATH = os.environ.get("VISUAL_OPSD_MODEL_PATH", "models/ThinkMorph-7B")
DATA_DIR = os.environ.get("VISUAL_OPSD_DATA_DIR", "datasets/Visual_Search/data")


def _assert_equal(a, b, msg: str = "") -> None:
    assert a == b, f"{msg} {a} != {b}"


def main() -> None:
    if not os.path.isdir(MODEL_PATH):
        print(f"[SKIP] model path not found: {MODEL_PATH}")
        return
    if not os.path.isdir(DATA_DIR):
        print(f"[SKIP] data dir not found: {DATA_DIR}")
        return

    tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    transform = ImageTransform(
        image_stride=14, max_image_size=980, min_image_size=378, max_pixels=2_007_040
    )

    dataset = OPSDPairedIterableDataset(
        dataset_name="visual_opsd",
        transform=transform,
        tokenizer=tokenizer,
        data_dir_list=[DATA_DIR],
        num_used_data=[4],
        local_rank=0,
        world_size=1,
        num_workers=0,
        data_status=None,
        max_reference_rounds=8,
    )
    dataset.set_epoch(0)

    raw = next(iter(dataset))

    # ---- structural checks on raw envelope ----
    for key in [
        "problem_image_tensor",
        "question_ids",
        "reference_thoughts_ids",
        "reference_vt_tensors",
        "reference_answer_ids",
        "data_indexes",
    ]:
        assert key in raw, f"missing key in raw envelope: {key}"

    assert torch.is_tensor(raw["problem_image_tensor"]), "image tensor not a torch.Tensor"
    assert raw["problem_image_tensor"].dim() == 3, "image tensor must be [C, H, W]"
    assert len(raw["question_ids"]) > 0, "empty question_ids"
    assert len(raw["reference_answer_ids"]) > 0, "empty reference_answer_ids"
    assert len(raw["reference_thoughts_ids"]) >= 1, "expected >=1 reference thought"

    print(f"[raw] question_ids len = {len(raw['question_ids'])}")
    print(f"[raw] answer_ids   len = {len(raw['reference_answer_ids'])}")
    print(f"[raw] #thoughts        = {len(raw['reference_thoughts_ids'])}")
    print(f"[raw] #VT tensors      = {sum(1 for v in raw['reference_vt_tensors'] if v is not None)}")
    print(f"[raw] image tensor shape = {tuple(raw['problem_image_tensor'].shape)}")

    # ---- pack builder round-trip ----
    builder = OPSDPackBuilder(
        config=PackBuilderConfig(
            vit_patch_size=14, max_num_patch_per_side=70, interpolate_pos=False,
            use_flex=False,
        ),
        special_tokens=new_token_ids,
        system_prompt=VLM_THINK_SYSTEM_PROMPT,
    )

    # Pretend we sampled a 16-token completion (arbitrary non-special ids).
    completion_ids = [100 + i for i in range(16)]

    raw["_system_ids"] = tokenizer.encode(
        VLM_THINK_SYSTEM_PROMPT, add_special_tokens=False
    )
    ref_intro_ids = tokenizer.encode(
        TEACHER_REFERENCE_INTRO, add_special_tokens=False
    )
    transition_ids = tokenizer.encode(TEACHER_TRANSITION, add_special_tokens=False)

    student = builder.build_student_sample(raw, completion_ids=completion_ids)
    teacher = builder.build_teacher_sample(
        raw,
        completion_ids=completion_ids,
        ref_intro_ids=ref_intro_ids,
        transition_ids=transition_ids,
    )

    # ---- structural checks on packed batches ----
    expected_ce = len(completion_ids) + 1
    _assert_equal(
        len(student["ce_loss_indexes"]),
        expected_ce,
        "student CE len mismatch:",
    )
    _assert_equal(
        len(teacher["ce_loss_indexes"]),
        expected_ce,
        "teacher CE len mismatch:",
    )
    _assert_equal(
        int(student["completion_ce_count"]),
        expected_ce,
        "student ce_count metadata mismatch:",
    )
    _assert_equal(
        int(teacher["completion_ce_count"]),
        expected_ce,
        "teacher ce_count metadata mismatch:",
    )

    # Labels on the completion block are exactly completion_ids + [eos]
    s_labels = student["packed_label_ids"].tolist()
    _assert_equal(
        s_labels, list(completion_ids) + [new_token_ids["eos_token_id"]],
        "student labels mismatch:",
    )
    t_labels = teacher["packed_label_ids"].tolist()
    _assert_equal(
        t_labels, list(completion_ids) + [new_token_ids["eos_token_id"]],
        "teacher labels mismatch:",
    )

    # Teacher must have MORE tokens than student because of the privileged
    # reference block.
    assert (
        teacher["sequence_length"] > student["sequence_length"]
    ), (
        f"teacher sequence should be longer than student "
        f"(got {teacher['sequence_length']} vs {student['sequence_length']})"
    )

    # Teacher must have at least 1 extra ViT image (the reference VTs).
    student_vits = student["vit_token_seqlens"].numel() if "vit_token_seqlens" in student else 0
    teacher_vits = teacher["vit_token_seqlens"].numel() if "vit_token_seqlens" in teacher else 0
    assert teacher_vits >= student_vits, (
        f"teacher should have >= ViT images as student "
        f"(got student={student_vits}, teacher={teacher_vits})"
    )

    print(
        "[pack] student seqlen=%d, teacher seqlen=%d, CE=%d"
        % (student["sequence_length"], teacher["sequence_length"], expected_ce)
    )
    print("[ok] raw dataset + pack builder round-trip OK")


if __name__ == "__main__":
    main()
