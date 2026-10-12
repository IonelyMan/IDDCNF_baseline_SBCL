"""Export the SBCL binary classifier's fc input and draw A/U diagnostics."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

from au_diagnostics.au_visualization import save_report, validate_features


ROOT = Path(os.path.abspath(__file__)).parent


def absolute_path(path):
    return str(Path(os.path.abspath(path)))


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"sample_id", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("manifest needs sample_id,label columns")
        rows = [(row["sample_id"], int(row["label"])) for row in reader]
    ids = [row[0] for row in rows]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError("manifest must be nonempty and have unique sample_id values")
    if any(label not in (0, 1) for _, label in rows):
        raise ValueError("manifest labels must be 0 or 1")
    return rows


def extract_from_checkpoint(args):
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    from torchvision.datasets import ImageFolder

    # Reuse exactly the existing second-stage checkpoint restoration, including
    # the NormedLinear variant. This does not alter the classifier's forward path.
    sys.path.insert(0, str(ROOT / "moco"))
    from infer import load_classifier

    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    data_root = Path(absolute_path(args.data_path))
    split_root = data_root / args.split
    dataset = ImageFolder(str(split_root), transform=transform)
    if dataset.classes != ["benign", "mal"]:
        raise ValueError(f"expected benign and mal directories in {split_root}")
    all_ids = [Path(path).relative_to(data_root).as_posix() for path, _ in dataset.samples]
    id_to_index = {sample_id: index for index, sample_id in enumerate(all_ids)}
    if len(id_to_index) != len(all_ids):
        raise ValueError("duplicate sample IDs in data split")
    if args.sample_manifest:
        requested = read_manifest(args.sample_manifest)
        missing = [sample_id for sample_id, _ in requested if sample_id not in id_to_index]
        if missing:
            raise ValueError(f"manifest references missing sample ID: {missing[0]}")
        indices = [id_to_index[sample_id] for sample_id, _ in requested]
        for (sample_id, label), index in zip(requested, indices):
            if dataset.targets[index] != label:
                raise ValueError(f"manifest label conflicts with directory for {sample_id}")
    else:
        indices = list(range(len(dataset)))
        if len(indices) > args.max_samples:
            rng = np.random.default_rng(args.seed)
            reserved = []
            for label in (0, 1):
                members = np.flatnonzero(np.asarray(dataset.targets) == label)
                keep = min(len(members), 2 if args.max_samples >= 4 else 1)
                reserved.extend(rng.choice(members, size=keep, replace=False).tolist())
            if len(reserved) > args.max_samples:
                reserved = reserved[:args.max_samples]
            remaining = np.setdiff1d(np.arange(len(dataset)), reserved)
            extra = rng.choice(remaining, size=args.max_samples - len(reserved), replace=False)
            indices = sorted(reserved + extra.tolist())
    if len(indices) < 2:
        raise ValueError("at least two diagnostic samples are required")
    ids = np.asarray([all_ids[i] for i in indices])
    labels = np.asarray([dataset.targets[i] for i in indices], dtype=np.int64)
    loader = DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = load_classifier(args.checkpoint, device)
    checkpoint_arch = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["arch"]
    model.eval()
    if not hasattr(model, "fc") or not isinstance(model.fc, torch.nn.Module):
        raise ValueError("checkpoint classifier has no final fc module")
    captured = []

    def capture_input(_module, inputs):
        if len(inputs) != 1 or inputs[0].ndim != 2:
            raise ValueError("final fc input is not a single [B,D] representation")
        captured.append(inputs[0].detach().cpu().numpy().copy())

    handle = model.fc.register_forward_pre_hook(capture_input)
    batches = []
    try:
        with torch.inference_mode():
            for images, _ in loader:
                captured.clear()
                logits = model(images.to(device, non_blocking=True))
                if logits.ndim != 2 or logits.shape != (len(images), 2):
                    raise ValueError("classifier output is not [B,2]")
                if len(captured) != 1 or captured[0].shape[0] != len(images):
                    raise ValueError("expected exactly one fc-input vector per sample")
                batches.append(captured[0])
    finally:
        handle.remove()
    features = np.concatenate(batches, axis=0)
    validate_features(features, labels, ids)
    return features, labels, ids, {
        "model": checkpoint_arch, "layer": "avgpool_flatten_before_fc",
        "split": args.split, "preprocessing": "RGB; Resize(256); CenterCrop(224); ToTensor; ImageNet normalize",
        "source": {"checkpoint": absolute_path(args.checkpoint),
                   "sample_manifest": absolute_path(args.sample_manifest) if args.sample_manifest else None,
                   "data_path": absolute_path(args.data_path),
                   "sampling_seed": None if args.sample_manifest else args.seed,
                   "max_samples": None if args.sample_manifest else args.max_samples},
    }


def main():
    parser = argparse.ArgumentParser(description="SBCL A/U seven-panel diagnostic")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", help="moco/linear_classify.py second-stage checkpoint")
    source.add_argument("--features_npz", help="saved features.npz for model-free redraw")
    parser.add_argument("--data_path", "--data", help="dataset root containing split/benign and split/mal")
    parser.add_argument("--split", default="val")
    parser.add_argument("--sample_manifest", help="CSV with sample_id,label; its rows are used in exact order")
    parser.add_argument("--max_samples", type=int, default=1024,
                        help="cap when no manifest is supplied; default 1024")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--align_alpha", type=float, default=2.0)
    parser.add_argument("--uniform_t", type=float, default=2.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0 or args.max_samples < 2:
        parser.error("batch_size must be positive; workers nonnegative; max_samples at least 2")
    if args.checkpoint:
        if not args.data_path:
            parser.error("--data_path is required with --checkpoint")
        if not Path(args.checkpoint).is_file():
            parser.error(f"checkpoint not found: {args.checkpoint}")
        features, labels, ids, context = extract_from_checkpoint(args)
    else:
        with np.load(args.features_npz, allow_pickle=False) as cache:
            features = cache["features"]
            labels = cache["labels"]
            ids = cache["sample_ids"]
        validate_features(features, labels, ids)
        old_meta = Path(absolute_path(args.features_npz)).parent / "metadata.json"
        metadata = json.loads(old_meta.read_text(encoding="utf-8")) if old_meta.exists() else {}
        context = {
            "model": metadata.get("model") or "Cached features",
            "layer": metadata.get("feature_layer") or "unspecified_cached_representation",
            "split": metadata.get("split"), "preprocessing": metadata.get("preprocessing"),
            "source": {"checkpoint": metadata.get("checkpoint"),
                       "features_npz": absolute_path(args.features_npz),
                       "sample_manifest": metadata.get("sample_manifest_source"),
                       "data_path": metadata.get("data_path"),
                       "sampling_seed": metadata.get("sampling_seed"),
                       "max_samples": metadata.get("max_samples")},
        }
    metrics = save_report(args.output_dir, features, labels, ids, context["model"],
                          context["layer"], context["source"], context["split"],
                          context["preprocessing"], args.align_alpha, args.uniform_t)
    print(f"Saved {len(labels)} samples to {absolute_path(args.output_dir)}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
