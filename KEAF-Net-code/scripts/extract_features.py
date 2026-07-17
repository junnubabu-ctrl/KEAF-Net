"""Pre-extract dual-stream features (Section 3.2).

Writes the per-image grid features and per-question token features consumed by
:class:`keaf_net.data.VQADataset`. Region (Faster R-CNN / bottom-up-attention)
features are produced by the standard detector dump and are expected at
``features/<split>/region/<image_id>.pt`` already, so this script only covers the
ViT grid stream and the BERT question stream.

    python scripts/extract_features.py --config configs/okvqa.yaml \
        --split train --images data/coco/train2014

Requires the ``extract`` extra (``pip install -e .[extract]``).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import argparse
import json
import os

import torch
from tqdm import tqdm

from keaf_net.config import load_config
from keaf_net.models.encoders import TextEncoder, VisualEncoder, patch_boxes


def _vit_transform(image_size: int = 224):
    from torchvision import transforms

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ViT grid and BERT question features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--images", required=True, help="directory of COCO images")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    from PIL import Image  # optional dependency, only needed for extraction

    cfg = load_config(args.config)
    out_root = os.path.join(cfg.data.feature_root, args.split)
    os.makedirs(os.path.join(out_root, "grid"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "question"), exist_ok=True)

    visual = VisualEncoder().to(args.device).eval()
    text = TextEncoder().to(args.device).eval()
    transform = _vit_transform(args.image_size)
    boxes = patch_boxes(args.image_size)

    with open(os.path.join(cfg.data.data_root, cfg.data.dataset, f"{args.split}.json")) as fh:
        records = json.load(fh)

    seen_images: set[str] = set()
    for rec in tqdm(records, desc=f"extract {args.split}"):
        image_id = str(rec["image_id"])
        if image_id not in seen_images:
            path = os.path.join(args.images, rec["image_file"])
            pixel = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(args.device)
            grid = visual(pixel).squeeze(0).cpu()
            torch.save({"feat": grid, "boxes": boxes}, os.path.join(out_root, "grid", f"{image_id}.pt"))
            seen_images.add(image_id)

        tokens, qcls = text([rec["question"]], device=args.device)
        torch.save({"tokens": tokens.squeeze(0).cpu(), "qcls": qcls.squeeze(0).cpu()},
                   os.path.join(out_root, "question", f"{rec['qid']}.pt"))

    print(f"extracted features for {len(seen_images)} images and {len(records)} questions")


if __name__ == "__main__":
    main()
