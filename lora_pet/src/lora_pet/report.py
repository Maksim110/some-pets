from __future__ import annotations

import json
from pathlib import Path

from .config import ProjectConfig, project_path
from .training import build_train_command, format_command


def write_experiment_card(cfg: ProjectConfig, output_path: str | Path = "reports/experiment_card.md") -> Path:
    target = project_path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    dataset_summary = _read_json(project_path(cfg.data.processed_dir) / "dataset_summary.json")
    quality_report = _read_json(project_path(cfg.data.processed_dir) / "quality_report.json")
    generation_manifest = _read_json(project_path(cfg.inference.output_dir) / "manifest.json")

    lines = [
        "# FLUX LoRA Experiment Card",
        "",
        "## Goal",
        "",
        (
            "Build a compact, reproducible DreamBooth LoRA pipeline for a FLUX text-to-image "
            "model: data preparation, parameter-efficient fine-tuning, inference, and lightweight "
            "quality checks."
        ),
        "",
        "## Setup",
        "",
        f"- Base model: `{cfg.model.base_model}`",
        f"- LoRA rank: `{cfg.training.rank}`",
        f"- Instance prompt: `{cfg.prompt.instance_prompt}`",
        f"- Training steps: `{cfg.training.max_train_steps}`",
        f"- Resolution: `{cfg.data.resolution}`",
        "",
        "## Dataset",
        "",
        f"- Accepted images: `{dataset_summary.get('accepted', 'not prepared')}`",
        f"- Skipped: `{dataset_summary.get('skipped', {})}`",
        f"- Quality summary: `{quality_report.get('summary', 'not evaluated')}`",
        "",
        "## Training Command",
        "",
        "```bash",
        format_command(build_train_command(cfg)),
        "```",
        "",
        "## Inference",
        "",
        f"- Samples directory: `{project_path(cfg.inference.output_dir)}`",
        f"- Generated samples in manifest: `{_safe_len(generation_manifest)}`",
        "",
        "## Resume Bullets",
        "",
        (
            "- Built a reproducible FLUX DreamBooth LoRA pipeline with custom data curation, "
            "PEFT training via Diffusers/Accelerate, and image generation evaluation."
        ),
        (
            "- Implemented dataset validation, duplicate filtering, configurable training runs, "
            "LoRA inference, and experiment-card reporting for qualitative analysis."
        ),
        "",
    ]

    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _read_json(path: Path):
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_len(value) -> int | str:
    if isinstance(value, list):
        return len(value)
    return "not generated"
