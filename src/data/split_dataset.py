"""
Split dataset items with the same filename stem across subfolders
into train/valid/test sets by percentage ratio.

Usage:
    python split_dataset.py --data-dir data --train-ratio 80 --valid-ratio 15 --test-ratio 5 --seed 42

This will create:
    data/train/<subfolder>/*.png
    data/valid/<subfolder>/*.png
    data/test/<subfolder>/*.png

and CSV files:
    data/train_ids.csv
    data/valid_ids.csv
    data/test_ids.csv

By default files are copied using shutil.copy2.
Use --overwrite to remove existing train/valid/test folders and id CSVs before splitting.
"""

from pathlib import Path
import argparse
import random
import shutil
import csv
import sys
import math


def find_subfolders(data_dir: Path):
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]

    # 기존 split 결과 폴더는 제외
    subdirs = [
        p for p in subdirs
        if p.name.lower() not in ("train", "valid", "test")
    ]

    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    good = []
    for d in subdirs:
        for f in d.iterdir():
            if f.is_file() and f.suffix.lower() in img_exts:
                good.append(d)
                break

    return sorted(good)


def collect_ids(subfolders):
    sets = []

    for d in subfolders:
        stems = {
            f.stem
            for f in d.iterdir()
            if f.is_file()
        }
        sets.append(stems)

    common = set.intersection(*sets) if sets else set()

    def sort_key(x):
        return int(x) if str(x).isdigit() else str(x)

    return sorted(common, key=sort_key)


def compute_split_sizes(total, train_ratio, valid_ratio, test_ratio):
    ratios = [train_ratio, valid_ratio, test_ratio]
    ratio_sum = sum(ratios)

    if ratio_sum <= 0:
        raise ValueError("Ratio sum must be positive.")

    if not math.isclose(ratio_sum, 100.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"train/valid/test ratios must sum to 100. "
            f"Current sum = {ratio_sum}"
        )

    raw_sizes = [
        total * train_ratio / 100.0,
        total * valid_ratio / 100.0,
        total * test_ratio / 100.0,
    ]

    # 우선 내림
    sizes = [int(math.floor(x)) for x in raw_sizes]

    # 내림 때문에 남은 개수는 소수점 부분이 큰 split부터 하나씩 배분
    remaining = total - sum(sizes)
    remainders = [
        (raw_sizes[i] - sizes[i], i)
        for i in range(3)
    ]
    remainders.sort(reverse=True)

    for _, idx in remainders[:remaining]:
        sizes[idx] += 1

    train_size, valid_size, test_size = sizes

    return train_size, valid_size, test_size


def copy_split(split_ids, subfolders, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for sf in subfolders:
        (out_dir / sf.name).mkdir(parents=True, exist_ok=True)

    copied = 0

    for sid in split_ids:
        for sf in subfolders:
            src_candidates = list(sf.glob(f"{sid}.*"))

            if not src_candidates:
                print(f"Warning: missing {sid} in {sf}")
                continue

            src = src_candidates[0]
            dst = out_dir / sf.name / src.name

            shutil.copy2(str(src), str(dst))
            copied += 1

    return copied


def write_csv(path: Path, ids):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id'])

        for sid in ids:
            writer.writerow([sid])


def remove_existing_split_outputs(data_dir: Path):
    for split_name in ("train", "valid", "test"):
        split_dir = data_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    for csv_name in ("train_ids.csv", "valid_ids.csv", "test_ids.csv"):
        csv_path = data_dir / csv_name
        if csv_path.exists():
            csv_path.unlink()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('data'),
        help='Path to data directory'
    )

    parser.add_argument(
        '--train-ratio',
        type=float,
        default=80.0,
        help='Train split ratio in percent'
    )

    parser.add_argument(
        '--valid-ratio',
        type=float,
        default=15.0,
        help='Validation split ratio in percent'
    )

    parser.add_argument(
        '--test-ratio',
        type=float,
        default=5.0,
        help='Test split ratio in percent'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Do not copy files, just print split plan'
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Remove existing train/valid/test folders and id CSVs before splitting'
    )

    args = parser.parse_args()

    data_dir = args.data_dir

    if not data_dir.exists():
        print(f"Error: data dir {data_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    if args.overwrite and not args.dry_run:
        remove_existing_split_outputs(data_dir)

    subfolders = find_subfolders(data_dir)

    if not subfolders:
        print(
            f"Error: no valid subfolders with images found in {data_dir}",
            file=sys.stderr
        )
        sys.exit(2)

    print("Found subfolders:")
    for s in subfolders:
        print("  -", s.name)

    ids = collect_ids(subfolders)
    total = len(ids)

    print(f"Found {total} complete items present in all subfolders")

    if total == 0:
        print("Error: no complete items found.", file=sys.stderr)
        sys.exit(2)

    train_size, valid_size, test_size = compute_split_sizes(
        total,
        args.train_ratio,
        args.valid_ratio,
        args.test_ratio,
    )

    print(
        f"Split ratio: "
        f"train={args.train_ratio:.2f}%, "
        f"valid={args.valid_ratio:.2f}%, "
        f"test={args.test_ratio:.2f}%"
    )

    print(
        f"Split size: "
        f"train={train_size}, "
        f"valid={valid_size}, "
        f"test={test_size}, "
        f"total={train_size + valid_size + test_size}"
    )

    random.seed(args.seed)

    ids_copy = ids.copy()
    random.shuffle(ids_copy)

    train_ids = ids_copy[:train_size]
    valid_ids = ids_copy[train_size:train_size + valid_size]
    test_ids = ids_copy[train_size + valid_size:train_size + valid_size + test_size]

    if args.dry_run:
        print("Dry run: no files will be copied.")
        print("Example train ids:", train_ids[:5])
        print("Example valid ids:", valid_ids[:5])
        print("Example test ids:", test_ids[:5])
        return

    train_out = data_dir / 'train'
    valid_out = data_dir / 'valid'
    test_out = data_dir / 'test'

    copied_train = copy_split(train_ids, subfolders, train_out)
    copied_valid = copy_split(valid_ids, subfolders, valid_out)
    copied_test = copy_split(test_ids, subfolders, test_out)

    write_csv(data_dir / 'train_ids.csv', train_ids)
    write_csv(data_dir / 'valid_ids.csv', valid_ids)
    write_csv(data_dir / 'test_ids.csv', test_ids)

    print(f"Copied: {copied_train} files to {train_out}")
    print(f"Copied: {copied_valid} files to {valid_out}")
    print(f"Copied: {copied_test} files to {test_out}")

    print(f"Wrote: {data_dir / 'train_ids.csv'}")
    print(f"Wrote: {data_dir / 'valid_ids.csv'}")
    print(f"Wrote: {data_dir / 'test_ids.csv'}")

    print("Done.")


if __name__ == '__main__':
    main()



'''
"""
Split dataset triples (same filename stem across subfolders) into train/valid sets.

Usage:
    python split_dataset.py --data-dir data --train-size 50 --valid-size 50 --seed 42

This will create:
    data/train/<subfolder>/*.png
    data/valid/<subfolder>/*.png
and CSV files:
    data/train_ids.csv
    data/valid_ids.csv

By default files are copied (shutil.copy2). If there are fewer complete triples than requested,
sizes will be adjusted automatically.
"""
from pathlib import Path
import argparse
import random
import shutil
import csv
import sys


def find_subfolders(data_dir: Path):
    # Find subdirectories containing image files, ignoring archive files
    subdirs = [p for p in data_dir.iterdir() if p.is_dir()]
    # Filter out common unwanted dirs like 'train' or 'valid' if they already exist
    subdirs = [p for p in subdirs if p.name.lower() not in ("train", "valid")] 
    # Keep only those that contain at least one file with an image extension
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    good = []
    for d in subdirs:
        for f in d.iterdir():
            if f.suffix.lower() in img_exts and f.is_file():
                good.append(d)
                break
    return sorted(good)


def collect_ids(subfolders):
    sets = []
    for d in subfolders:
        stems = {f.stem for f in d.iterdir() if f.is_file()}
        sets.append(stems)
    # intersection across all
    common = set.intersection(*sets) if sets else set()
    return sorted(common)


def copy_split(ids, split_ids, subfolders, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    # create subfolders inside out_dir to match original structure
    for sf in subfolders:
        (out_dir / sf.name).mkdir(parents=True, exist_ok=True)
    copied = 0
    for sid in split_ids:
        for sf in subfolders:
            src_candidates = list(sf.glob(f"{sid}.*"))
            if not src_candidates:
                # no file with that stem in this subfolder; skip and warn
                print(f"Warning: missing {sid} in {sf}")
                continue
            # choose the first candidate (preserves original extension)
            src = src_candidates[0]
            dst = out_dir / sf.name / src.name
            shutil.copy2(str(src), str(dst))
            copied += 1
    return copied


def write_csv(path: Path, ids):
    with path.open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['id'])
        for i in ids:
            w.writerow([i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('data'), help='Path to data directory')
    parser.add_argument('--train-size', type=int, default=50, help='Number of items for train')
    parser.add_argument('--valid-size', type=int, default=50, help='Number of items for valid')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Do not copy files, just print plan')
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.exists():
        print(f"Error: data dir {data_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    subfolders = find_subfolders(data_dir)
    if not subfolders:
        print(f"Error: no valid subfolders with images found in {data_dir}", file=sys.stderr)
        sys.exit(2)

    print("Found subfolders:")
    for s in subfolders:
        print("  -", s.name)

    ids = collect_ids(subfolders)
    total = len(ids)
    print(f"Found {total} complete items (present in all subfolders)")

    wanted = args.train_size + args.valid_size
    if total < wanted:
        print(f"Warning: requested {wanted} items but only {total} available. Adjusting sizes.")
        # make train/valid roughly half-half while keeping requested ratio if possible
        half = total // 2
        train_size = min(args.train_size, half)
        valid_size = total - train_size
    else:
        train_size = args.train_size
        valid_size = args.valid_size

    random.seed(args.seed)
    ids_copy = ids.copy()
    random.shuffle(ids_copy)
    train_ids = ids_copy[:train_size]
    valid_ids = ids_copy[train_size:train_size+valid_size]

    print(f"Train: {len(train_ids)} items, Valid: {len(valid_ids)} items")

    if args.dry_run:
        print("Dry run: no files will be copied. Example train ids:")
        print(train_ids[:5])
        return

    # perform copy
    train_out = data_dir / 'train'
    valid_out = data_dir / 'valid'
    copied_train = copy_split(ids, train_ids, subfolders, train_out)
    copied_valid = copy_split(ids, valid_ids, subfolders, valid_out)

    write_csv(data_dir / 'train_ids.csv', train_ids)
    write_csv(data_dir / 'valid_ids.csv', valid_ids)

    print(f"Copied: {copied_train} files to {train_out}")
    print(f"Copied: {copied_valid} files to {valid_out}")
    print("Done.")


if __name__ == '__main__':
    main()
'''