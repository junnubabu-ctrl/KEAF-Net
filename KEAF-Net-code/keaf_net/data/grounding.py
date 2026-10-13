"""Explicit detector, token-span and coordinate contracts for cached evidence.

These repair index-space ambiguities in the legacy cache. Cache version 2 must
be regenerated; an old phrase ordinal cannot safely be guessed into a BERT token.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence

import torch


def question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def region_pairs_to_visual(pairs: Sequence[tuple[int, int]], num_grid: int,
                           num_regions: int) -> torch.Tensor:
    """Translate detector row IDs into [grid; region] visual node IDs."""
    if any(i < 0 or i >= num_regions for i, _ in pairs):
        raise ValueError("detector link index is outside the region feature rows")
    return torch.tensor([(num_grid + i, k) for i, k in pairs], dtype=torch.long).reshape(-1, 2)


def phrase_pairs_to_tokens(pairs: Sequence[tuple[int, int]],
                           spans: Sequence[tuple[int, int]],
                           offsets: torch.Tensor) -> torch.Tensor:
    """Map each linked character span to every overlapping BERT WordPiece.

    Special/padding offsets (0,0) and truncated-away spans create no edge.
    Character offsets must come from the tokenizer used for the cached states.
    """
    if offsets.ndim != 2 or offsets.size(1) != 2:
        raise ValueError("token offsets must have shape [tokens,2]")
    linked: set[tuple[int, int]] = set()
    for phrase, knowledge in pairs:
        if phrase < 0 or phrase >= len(spans):
            raise ValueError("phrase link index is outside the noun-phrase spans")
        start, end = spans[phrase]
        if start < 0 or end <= start:
            raise ValueError("noun phrase must have a nonempty character span")
        mask = (offsets[:, 1] > start) & (offsets[:, 0] < end) & (offsets[:, 1] > offsets[:, 0])
        linked.update((token, knowledge) for token in mask.nonzero().flatten().tolist())
    return torch.tensor(sorted(linked), dtype=torch.long).reshape(-1, 2)


def boxes_in_grid_frame(region: dict, grid: dict) -> torch.Tensor:
    """Put detector XYXY boxes into the grid's resized pixel frame.

    External detector dumps must declare processed_pixels, original_pixels or
    normalized_xyxy, and their image_size where required. Frame metadata is
    required; box maxima do not establish which coordinate system was used.
    """
    if (grid.get("box_coordinate_frame") != "processed_pixels" or "image_size" not in grid
            or grid.get("spatial_transform") != "resize_only"):
        raise ValueError("grid boxes require processed_pixels and image_size metadata; rebuild cache")
    target = torch.as_tensor(grid["image_size"], dtype=region["boxes"].dtype)
    if target.shape != (2,) or (target <= 0).any():
        raise ValueError("grid image_size must be positive [width,height]")
    boxes = region["boxes"]
    if boxes.ndim != 2 or boxes.size(1) != 4 or not torch.isfinite(boxes).all():
        raise ValueError("detector boxes must be finite [regions,4] XYXY")
    frame = region.get("box_coordinate_frame")
    if frame in {"normalized_xyxy", "processed_pixels"} and region.get("spatial_transform") != "resize_only":
        raise ValueError("processed/normalized detector boxes require resize_only geometry; crops/letterboxes are unsupported")
    if frame == "normalized_xyxy":
        source = target.new_ones(2)
    elif frame in {"original_pixels", "processed_pixels"}:
        if "image_size" not in region:
            raise ValueError("detector pixel boxes require image_size [width,height]")
        source = torch.as_tensor(region["image_size"], dtype=boxes.dtype)
        if source.shape != (2,) or (source <= 0).any():
            raise ValueError("detector image_size must be positive [width,height]")
        if frame == "original_pixels":
            native = torch.as_tensor(grid.get("source_image_size", []), dtype=boxes.dtype)
            if native.shape != (2,) or not torch.equal(source, native):
                raise ValueError("detector original image size does not match grid provenance")
    else:
        raise ValueError("detector coordinate frame is missing or unsupported; annotate the source dump")
    limit = source.repeat(2)
    if (boxes < 0).any() or (boxes > limit).any() or (boxes[:, 2:] < boxes[:, :2]).any():
        raise ValueError("detector boxes are outside the declared coordinate frame")
    return boxes * (target / source).repeat(2)


def validate_cache_alignment(knowledge: dict, grid: dict, region: dict,
                             question: dict, text: str) -> None:
    """Reject stale or legacy links rather than attach them to unrelated nodes."""
    alignment = knowledge.get("alignment", {})
    expected = {"version": 2, "num_grid": grid["feat"].size(0),
                "num_regions": region["feat"].size(0),
                "num_tokens": question["tokens"].size(0), "question_sha256": question_sha256(text)}
    if any(alignment.get(key) != value for key, value in expected.items()):
        raise ValueError("knowledge cache alignment is missing/stale; regenerate version-2 retrieval caches")
    if question.get("question_sha256") != expected["question_sha256"]:
        raise ValueError("question feature cache does not match the annotation text")
    tokenizer = question.get("tokenizer_name")
    if not tokenizer or alignment.get("tokenizer_name") != tokenizer:
        raise ValueError("knowledge and question caches require the same recorded tokenizer identity")
    if knowledge.get("vk_index_space") != "visual_grid_then_region" or knowledge.get("tk_index_space") != "bert_token":
        raise ValueError("knowledge cache must declare visual and BERT-token index spaces")
