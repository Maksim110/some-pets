from __future__ import annotations

import json
import shutil
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import ProjectConfig, project_path


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass
class DatasetItem:
    output_path: str
    source_path: str
    original_size: tuple[int, int]
    processed_size: tuple[int, int]
    avg_hash: str
    caption: str


@dataclass
class DatasetReport:
    accepted: list[DatasetItem] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def download_urls(urls_file: str | Path, raw_dir: str | Path) -> list[Path]:
    raw_path = project_path(raw_dir)
    raw_path.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for idx, url in enumerate(_read_nonempty_lines(project_path(urls_file)), start=1):
        suffix = _suffix_from_url(url) or ".jpg"
        target = raw_path / f"url_{idx:04d}{suffix}"
        with urllib.request.urlopen(url, timeout=30) as response:
            with target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
        downloaded.append(target)

    return downloaded


def prepare_dataset(
    cfg: ProjectConfig,
    source_dir: str | Path | None = None,
    urls_file: str | Path | None = None,
    caption: str | None = None,
    force: bool = False,
) -> DatasetReport:
    if urls_file:
        download_urls(urls_file, cfg.data.raw_dir)

    source_path = project_path(source_dir or cfg.data.raw_dir)
    processed_path = project_path(cfg.data.processed_dir)
    instance_path = project_path(cfg.data.instance_dir)

    if force and processed_path.exists():
        shutil.rmtree(processed_path)

    instance_path.mkdir(parents=True, exist_ok=True)
    report = DatasetReport()
    seen_hashes: list[str] = []
    caption_text = caption or cfg.prompt.instance_prompt

    candidates = list(iter_image_files(source_path))
    for image_path in candidates:
        if cfg.data.max_images and len(report.accepted) >= cfg.data.max_images:
            report.skip("over_max_images")
            continue

        try:
            with Image.open(image_path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                original_size = image.size
                if min(original_size) < cfg.data.min_size:
                    report.skip("too_small")
                    continue

                image = _fit_image(image, cfg.data.resolution, cfg.data.center_crop)
                avg_hash = average_hash(image)
        except (OSError, UnidentifiedImageError):
            report.skip("unreadable")
            continue

        if any(hamming_distance(avg_hash, previous) <= cfg.data.duplicate_hamming_threshold for previous in seen_hashes):
            report.skip("duplicate")
            continue

        seen_hashes.append(avg_hash)
        output_name = f"{len(report.accepted) + 1:04d}_{cfg.prompt.token}.jpg"
        output_path = instance_path / output_name
        image.save(output_path, quality=95, optimize=True)
        report.accepted.append(
            DatasetItem(
                output_path=str(output_path),
                source_path=str(image_path),
                original_size=original_size,
                processed_size=image.size,
                avg_hash=avg_hash,
                caption=caption_text,
            )
        )

    write_dataset_artifacts(cfg, report)
    return report


def iter_image_files(path: str | Path) -> Iterable[Path]:
    root = project_path(path)
    if not root.exists():
        return []
    return sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    )


def write_dataset_artifacts(cfg: ProjectConfig, report: DatasetReport) -> None:
    processed_path = project_path(cfg.data.processed_dir)
    processed_path.mkdir(parents=True, exist_ok=True)

    metadata_path = processed_path / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as handle:
        for item in report.accepted:
            handle.write(json.dumps(item.__dict__, ensure_ascii=False) + "\n")

    summary = {
        "accepted": len(report.accepted),
        "skipped": report.skipped,
        "instance_dir": str(project_path(cfg.data.instance_dir)),
        "resolution": cfg.data.resolution,
        "prompt": cfg.prompt.instance_prompt,
    }
    (processed_path / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (processed_path / "dataset_report.md").write_text(
        render_dataset_report(cfg, report),
        encoding="utf-8",
    )


def render_dataset_report(cfg: ProjectConfig, report: DatasetReport) -> str:
    lines = [
        "# Dataset Report",
        "",
        f"- Project: `{cfg.project.name}`",
        f"- Accepted images: `{len(report.accepted)}`",
        f"- Resolution: `{cfg.data.resolution}`",
        f"- Instance prompt: `{cfg.prompt.instance_prompt}`",
        f"- Instance directory: `{project_path(cfg.data.instance_dir)}`",
        "",
        "## Skipped",
        "",
    ]
    if report.skipped:
        lines.extend(f"- {reason}: `{count}`" for reason, count in sorted(report.skipped.items()))
    else:
        lines.append("- none")

    lines.extend(["", "## Images", ""])
    for item in report.accepted:
        lines.append(
            f"- `{Path(item.output_path).name}` from `{Path(item.source_path).name}` "
            f"{item.original_size} -> {item.processed_size}"
        )
    lines.append("")
    return "\n".join(lines)


def average_hash(image: Image.Image, hash_size: int = 8) -> str:
    small = ImageOps.grayscale(image).resize((hash_size, hash_size), Image.Resampling.LANCZOS)
    pixels = np.asarray(small, dtype=np.float32)
    mean = float(pixels.mean())
    bits = pixels > mean
    return "".join("1" if value else "0" for value in bits.flatten())


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("hashes must have equal length")
    return sum(a != b for a, b in zip(left, right))


def _fit_image(image: Image.Image, resolution: int, center_crop: bool) -> Image.Image:
    if center_crop:
        return ImageOps.fit(
            image,
            (resolution, resolution),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
    return ImageOps.pad(
        image,
        (resolution, resolution),
        method=Image.Resampling.LANCZOS,
        color=(255, 255, 255),
        centering=(0.5, 0.5),
    )


def _read_nonempty_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _suffix_from_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in SUPPORTED_IMAGE_SUFFIXES:
        return suffix
    return ""
