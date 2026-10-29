#!/usr/bin/env python
"""
Generate the train.txt / val.txt this repo's Cityscapes loader expects.

dataset/cityscapes.py does

    open(root + list_path)                                  # the list file
    cv2.imread(os.path.join(root, 'cityscapes', item[...]))  # each entry

so the list files live in `root` (the PARENT of the cityscapes dir) and every
entry is relative to root/cityscapes/. An mmseg-style checkout has no such
files, which is the only thing standing between that layout and this loader.

LABELS ARE *_gtFine_labelIds.png, NOT labelTrainIds.png. Cityscapes.convert_label
maps raw ids 0-33 through label_mapping itself; hand it a trainIds file and the
mapping is applied twice, which silently relabels the ground truth instead of
raising.

    python scripts/make_lists.py --root D:/master-thesis/local/data
"""
from __future__ import annotations

import argparse
from pathlib import Path


def build(root: Path, split: str) -> list[str]:
    img_dir = root / "cityscapes" / "leftImg8bit" / split
    lbl_dir = root / "cityscapes" / "gtFine" / split
    if not img_dir.is_dir():
        raise SystemExit(f"missing {img_dir}")

    lines, missing = [], []
    for img in sorted(img_dir.rglob("*_leftImg8bit.png")):
        city = img.parent.name
        stem = img.name[: -len("_leftImg8bit.png")]
        lbl = lbl_dir / city / f"{stem}_gtFine_labelIds.png"
        if not lbl.is_file():
            missing.append(lbl)
            continue
        # Forward slashes: os.path.join on Windows accepts them, and the file
        # then reads identically on the cluster.
        lines.append(f"leftImg8bit/{split}/{city}/{img.name} "
                     f"gtFine/{split}/{city}/{lbl.name}")

    if missing:
        raise SystemExit(f"{len(missing)} images have no labelIds label, "
                         f"first: {missing[0]}")
    return lines


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="parent of the cityscapes/ directory")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    a = p.parse_args()

    root = Path(a.root)
    for split in a.splits:
        lines = build(root, split)
        out = root / f"{split}.txt"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{out}  {len(lines)} entries")


if __name__ == "__main__":
    main()
