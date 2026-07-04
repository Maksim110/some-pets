from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from .config import ProjectConfig, project_path
from .dataset import SUPPORTED_IMAGE_SUFFIXES, average_hash, hamming_distance, iter_image_files


@dataclass
class ImageStats:
    path: str
    width: int
    height: int
    brightness_mean: float
    brightness_std: float
    entropy: float
    avg_hash: str


def evaluate_dataset(cfg: ProjectConfig) -> dict[str, object]:
    image_paths = list(iter_image_files(cfg.data.instance_dir))
    stats = [_image_stats(path) for path in image_paths]
    report = _summarize_stats(stats)
    report["duplicate_pairs"] = _duplicate_pairs(stats, cfg.data.duplicate_hamming_threshold)
    report["source"] = str(project_path(cfg.data.instance_dir))
    report["expected_resolution"] = cfg.data.resolution

    output_dir = project_path(cfg.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_report(output_dir / "quality_report", report, stats)
    return report


def evaluate_generations(cfg: ProjectConfig, images_dir: str | Path | None = None) -> dict[str, object]:
    root = project_path(images_dir or cfg.inference.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )
    stats = [_image_stats(path) for path in image_paths]
    report = _summarize_stats(stats)
    report["duplicate_pairs"] = _duplicate_pairs(stats, cfg.data.duplicate_hamming_threshold)
    report["source"] = str(root)

    _write_report(root / "generation_quality", report, stats)
    return report


def _image_stats(path: Path) -> ImageStats:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
        return ImageStats(
            path=str(path),
            width=image.width,
            height=image.height,
            brightness_mean=float(gray.mean()),
            brightness_std=float(gray.std()),
            entropy=float(image.entropy()),
            avg_hash=average_hash(image),
        )


def _summarize_stats(stats: list[ImageStats]) -> dict[str, object]:
    if not stats:
        return {
            "count": 0,
            "brightness_mean": None,
            "brightness_std": None,
            "entropy_mean": None,
            "min_width": None,
            "min_height": None,
        }

    return {
        "count": len(stats),
        "brightness_mean": float(np.mean([item.brightness_mean for item in stats])),
        "brightness_std": float(np.mean([item.brightness_std for item in stats])),
        "entropy_mean": float(np.mean([item.entropy for item in stats])),
        "min_width": min(item.width for item in stats),
        "min_height": min(item.height for item in stats),
    }


def _duplicate_pairs(stats: list[ImageStats], threshold: int) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for left_idx, left in enumerate(stats):
        for right in stats[left_idx + 1 :]:
            distance = hamming_distance(left.avg_hash, right.avg_hash)
            if distance <= threshold:
                pairs.append(
                    {
                        "left": left.path,
                        "right": right.path,
                        "hamming_distance": distance,
                    }
                )
    return pairs


def _write_report(prefix: Path, report: dict[str, object], stats: list[ImageStats]) -> None:
    json_path = prefix.with_suffix(".json")
    markdown_path = prefix.with_suffix(".md")

    payload = {
        "summary": report,
        "images": [asdict(item) for item in stats],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(_render_markdown(report, stats), encoding="utf-8")


def _render_markdown(report: dict[str, object], stats: list[ImageStats]) -> str:
    lines = [
        "# Quality Report",
        "",
        f"- Image count: `{report['count']}`",
        f"- Mean brightness: `{report['brightness_mean']}`",
        f"- Mean brightness std: `{report['brightness_std']}`",
        f"- Mean entropy: `{report['entropy_mean']}`",
        f"- Duplicate pairs: `{len(report.get('duplicate_pairs', []))}`",
        "",
        "## Images",
        "",
    ]
    for item in stats:
        lines.append(
            f"- `{Path(item.path).name}`: {item.width}x{item.height}, "
            f"brightness={item.brightness_mean:.2f}, entropy={item.entropy:.2f}"
        )
    lines.append("")
    return "\n".join(lines)
