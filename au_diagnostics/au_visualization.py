"""High dimensional alignment/uniformity and the seven-panel diagnostic figure."""

import csv
import json
from importlib.metadata import version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
try:
    matplotlib.__version__
except OSError:
    # Some restricted Windows installations deny realpath on site-packages.
    matplotlib.__version__ = version("matplotlib")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import logsumexp


def validate_features(features, labels, sample_ids):
    z = np.asarray(features)
    y = np.asarray(labels)
    ids = np.asarray(sample_ids)
    real_numeric = np.issubdtype(z.dtype, np.integer) or np.issubdtype(z.dtype, np.floating)
    if z.ndim != 2 or min(z.shape) < 2 or not real_numeric:
        raise ValueError("features must be a numeric [N,D] array with N,D >= 2")
    if y.ndim != 1 or ids.ndim != 1 or len(z) != len(y) or len(z) != len(ids):
        raise ValueError("features, labels and sample_ids must have matching row counts")
    if not np.isfinite(z).all():
        raise ValueError("features contain NaN or Inf")
    if not np.issubdtype(y.dtype, np.integer) or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must contain only integer 0 (benign) or 1 (malicious)")
    ids = ids.astype(str)
    if np.any(ids == "") or len(set(ids.tolist())) != len(ids):
        raise ValueError("sample_ids must be nonempty and unique")
    return z, y.astype(np.int64), ids


def compute_diagnostics(features, labels, alpha=2.0, uniform_t=2.0):
    if alpha <= 0 or uniform_t <= 0:
        raise ValueError("align_alpha and uniform_t must be positive")
    z = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels)
    norms = np.linalg.norm(z, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("feature norms overflow; check feature scale")
    u = z / np.maximum(norms[:, None], 1e-12)
    distances = pdist(u, metric="euclidean")
    squared = pdist(u, metric="sqeuclidean")
    upper = np.triu_indices(len(y), k=1)
    same_mask = y[upper[0]] == y[upper[1]]
    class_alignment = {}
    valid_anchors = 0
    weighted_sum = 0.0
    for label, name in ((0, "benign"), (1, "malicious")):
        count = int(np.sum(y == label))
        if count < 2:
            class_alignment[name] = None
            continue
        pairs = (y[upper[0]] == label) & (y[upper[1]] == label)
        # Every within-class pair contributes to two anchors with the same denominator.
        value = float(np.mean(distances[pairs] ** alpha))
        class_alignment[name] = value
        valid_anchors += count
        weighted_sum += count * value
    alignment = weighted_sum / valid_anchors if valid_anchors else None
    uniformity = float(logsumexp(-uniform_t * squared) - np.log(len(squared))) if len(squared) else None
    metrics = {
        "alignment": alignment,
        "uniformity": uniformity,
        "exp_uniformity": float(np.exp(uniformity)) if uniformity is not None else None,
        "benign_alignment": class_alignment["benign"],
        "malicious_alignment": class_alignment["malicious"],
        "sample_count": len(y),
        "benign_count": int(np.sum(y == 0)),
        "malicious_count": int(np.sum(y == 1)),
        "valid_alignment_anchors": valid_anchors,
        "pair_count": len(distances),
        "same_class_pair_count": int(np.sum(same_mask)),
        "different_class_pair_count": int(np.sum(~same_mask)),
        "zero_vector_count": int(np.sum(norms == 0)),
    }
    return u, distances, same_mask, metrics


def project_circle(u, labels):
    _, _, vt = np.linalg.svd(u, full_matrices=False)
    basis = vt[:2].T.copy()
    for col in range(2):
        if basis[np.argmax(np.abs(basis[:, col])), col] < 0:
            basis[:, col] *= -1
    projected = u @ basis
    projected_norm = np.linalg.norm(projected, axis=1)
    valid = projected_norm > 1e-12
    circle = projected / np.maximum(projected_norm[:, None], 1e-12)
    counts = {
        "projection_degenerate_count": int(np.sum(~valid)),
        "benign_projection_degenerate_count": int(np.sum((labels == 0) & ~valid)),
        "malicious_projection_degenerate_count": int(np.sum((labels == 1) & ~valid)),
        "projection_plottable_count": int(np.sum(valid)),
    }
    return basis, circle, valid, counts


def _circle_density(points, grid, bandwidth=0.10, chunk_size=256):
    values = np.zeros((len(grid), len(grid)), dtype=np.float64)
    if len(points) == 0:
        return values
    xx, yy = np.meshgrid(grid, grid)
    for start in range(0, len(points), chunk_size):
        part = points[start:start + chunk_size]
        distance2 = (xx[:, :, None] - part[:, 0]) ** 2 + (yy[:, :, None] - part[:, 1]) ** 2
        values += np.exp(-distance2 / (2 * bandwidth ** 2)).sum(axis=2)
    return values / (len(points) * 2 * np.pi * bandwidth ** 2)


def _angle_density(points, grid, kappa=8.0):
    if len(points) == 0:
        return None
    angles = np.arctan2(points[:, 1], points[:, 0])
    return np.exp(kappa * np.cos(grid[:, None] - angles)).mean(axis=1) / (2 * np.pi * np.i0(kappa))


def plot_report(output_dir, u, labels, distances, same_mask, metrics, model_name, layer_name):
    basis, circle, valid, counts = project_circle(u, labels)
    metrics.update(counts)
    fig = plt.figure(figsize=(15, 5), layout="constrained")
    grid = fig.add_gridspec(2, 4, height_ratios=[3, 1])
    distance_ax = fig.add_subplot(grid[:, 0])
    bins = np.linspace(0, 2, 31)
    for mask, color, name in ((same_mask, "#2678ad", "Same class"),
                              (~same_mask, "#dd8452", "Different class")):
        if np.any(mask):
            distance_ax.hist(distances[mask], bins=bins, density=True, alpha=0.45,
                             color=color, label=name)
    if np.any(same_mask):
        distance_ax.axvline(float(np.mean(distances[same_mask])), color="#2678ad",
                            linestyle="--", label="Same-class pair mean")
    distance_ax.set(xlim=(0, 2), xlabel="Original-space L2 distance", ylabel="Pair density",
                    title="Alignment: pair distances")
    if distance_ax.get_legend_handles_labels()[0]:
        distance_ax.legend(fontsize=8)
    xy_grid = np.linspace(-1.2, 1.2, 80)
    theta_grid = np.linspace(-np.pi, np.pi, 256)
    circle_theta = np.linspace(0, 2 * np.pi, 256)
    groups = ((np.ones(len(labels), dtype=bool), f"All samples (n={len(labels)})"),
              (labels == 0, f"Benign (0) (n={metrics['benign_count']})"),
              (labels == 1, f"Malicious (1) (n={metrics['malicious_count']})"))
    background = np.asarray(plt.get_cmap("YlGnBu")(0.0))
    class_colors = ((1, "#c84e2f"), (0, "#2e7d32"))
    for col, (mask, title) in enumerate(groups, start=1):
        points = circle[mask & valid]
        top = fig.add_subplot(grid[0, col])
        top.set_facecolor(background)
        if len(points):
            if col == 1:
                # Each class has its own KDE scale; draw green last for natural overlap.
                for class_id, color in class_colors:
                    class_points = circle[valid & (labels == class_id)]
                    if len(class_points):
                        shades = ListedColormap([to_rgba(color, alpha)
                                                 for alpha in np.linspace(0, 0.85, 12)])
                        top.contourf(xy_grid, xy_grid, _circle_density(class_points, xy_grid),
                                     12, cmap=shades, zorder=2 if class_id == 0 else 1)
            else:
                color = "#2e7d32" if col == 2 else "#c84e2f"
                target = np.asarray(to_rgba(color))
                shades = ListedColormap([(1 - weight) * background + weight * target
                                         for weight in np.linspace(0, 1, 12)])
                top.contourf(xy_grid, xy_grid, _circle_density(points, xy_grid),
                             12, cmap=shades)
        else:
            top.text(0, 0, "No valid points", ha="center", va="center")
        top.plot(np.cos(circle_theta), np.sin(circle_theta), color="gray", linewidth=1)
        top.set(xlim=(-1.2, 1.2), ylim=(-1.2, 1.2), xticks=[-1, 0, 1],
                yticks=[-1, 0, 1], title=title)
        top.set_aspect("equal")
        bottom = fig.add_subplot(grid[1, col])
        density = _angle_density(points, theta_grid)
        if density is None:
            bottom.text(0, 0.5, "No valid points", ha="center", va="center",
                        transform=bottom.transAxes)
        else:
            bottom.plot(theta_grid, density, color="#246a9b")
            bottom.fill_between(theta_grid, density, alpha=0.20, color="#246a9b")
        bottom.axhline(1 / (2 * np.pi), color="gray", linestyle="--", linewidth=1)
        bottom.set(xlim=(-np.pi, np.pi), ylim=(0, None), xticks=[-np.pi, 0, np.pi],
                   xticklabels=["-pi", "0", "pi"], xlabel="Angle (radians)", ylabel="Density")
    def fmt(value):
        return "undefined" if value is None else f"{value:.4f}"
    fig.suptitle(f"{model_name} | Representation={layer_name} | D={u.shape[1]}\n"
                 f"high-D alignment={fmt(metrics['alignment'])}, uniformity={fmt(metrics['uniformity'])}\n"
                 "Circle/angle panels: shared SVD projection for display only", fontsize=11)
    fig.savefig(output_dir / "representation.png", dpi=180)
    fig.savefig(output_dir / "representation.svg")
    plt.close(fig)
    return basis


def save_report(output_dir, features, labels, sample_ids, model_name, layer_name,
                source, split, preprocessing, alpha=2.0, uniform_t=2.0):
    z, y, ids = validate_features(features, labels, sample_ids)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    u, distances, same_mask, metrics = compute_diagnostics(z, y, alpha, uniform_t)
    basis = plot_report(output_dir, u, y, distances, same_mask, metrics, model_name, layer_name)
    if metrics["zero_vector_count"]:
        print(f"Warning: {metrics['zero_vector_count']} zero vectors have no sphere direction.")
    np.savez_compressed(output_dir / "features.npz", features=z, labels=y, sample_ids=ids)
    np.save(output_dir / "projection_basis.npy", basis)
    with (output_dir / "sample_manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "label"])
        writer.writerows(zip(ids, y))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(list(metrics))
        writer.writerow(list(metrics.values()))
    metadata = {
        "model": model_name, "checkpoint": source.get("checkpoint"), "source_features_npz": source.get("features_npz"),
        "split": split, "feature_layer": layer_name, "dimension": z.shape[1],
        "preprocessing": preprocessing, "align_alpha": alpha, "uniform_t": uniform_t,
        "metric_space": "original dimension, row L2 normalized; no centering",
        "projection": "non-centered SVD of all diagnostic rows; one shared basis for all three panels; display only",
        "circle_bandwidth": 0.10, "angle_kappa": 8.0, "distance_bins": "linspace(0,2,31)",
        "sample_manifest_source": source.get("sample_manifest"),
        "data_path": source.get("data_path"),
        "sampling_seed": source.get("sampling_seed"),
        "max_samples": source.get("max_samples"),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metrics
