from __future__ import annotations

import subprocess
import sys
import urllib.request
from pathlib import Path

from .config import ProjectConfig, project_path


DIFFUSERS_RAW_BASE = "https://raw.githubusercontent.com/huggingface/diffusers/{ref}/examples/dreambooth"
TRAINER_FILES = [
    "train_dreambooth_lora_flux.py",
    "requirements_flux.txt",
]


def fetch_trainer(ref: str = "main", target_root: str | Path = "third_party/diffusers") -> list[Path]:
    target_dir = project_path(target_root) / "examples" / "dreambooth"
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for filename in TRAINER_FILES:
        url = f"{DIFFUSERS_RAW_BASE.format(ref=ref)}/{filename}"
        target = target_dir / filename
        urllib.request.urlretrieve(url, target)
        downloaded.append(target)

    return downloaded


def build_train_command(cfg: ProjectConfig) -> list[str]:
    trainer_script = project_path(cfg.training.trainer_script)
    validation_prompt = _first_validation_prompt(cfg)

    command = [
        "accelerate",
        "launch",
        str(trainer_script),
        "--pretrained_model_name_or_path",
        cfg.model.base_model,
        "--instance_data_dir",
        str(project_path(cfg.data.instance_dir)),
        "--output_dir",
        str(project_path(cfg.training.output_dir)),
        "--mixed_precision",
        cfg.training.mixed_precision,
        "--instance_prompt",
        cfg.prompt.instance_prompt,
        "--resolution",
        str(cfg.data.resolution),
        "--train_batch_size",
        str(cfg.training.train_batch_size),
        "--guidance_scale",
        str(cfg.training.guidance_scale),
        "--gradient_accumulation_steps",
        str(cfg.training.gradient_accumulation_steps),
        "--optimizer",
        cfg.training.optimizer,
        "--learning_rate",
        str(cfg.training.learning_rate),
        "--lr_scheduler",
        cfg.training.lr_scheduler,
        "--lr_warmup_steps",
        str(cfg.training.lr_warmup_steps),
        "--max_train_steps",
        str(cfg.training.max_train_steps),
        "--validation_prompt",
        validation_prompt,
        "--validation_epochs",
        str(cfg.training.validation_epochs),
        "--num_validation_images",
        str(cfg.training.num_validation_images),
        "--checkpointing_steps",
        str(cfg.training.checkpointing_steps),
        "--rank",
        str(cfg.training.rank),
        "--lora_alpha",
        str(cfg.training.lora_alpha),
        "--seed",
        str(cfg.project.seed),
        "--report_to",
        cfg.training.report_to,
    ]

    if cfg.training.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if cfg.training.cache_latents:
        command.append("--cache_latents")
    if cfg.training.allow_tf32:
        command.append("--allow_tf32")
    if cfg.training.use_8bit_adam:
        command.append("--use_8bit_adam")
    if cfg.data.center_crop:
        command.append("--center_crop")
    if cfg.training.lora_layers:
        command.extend(["--lora_layers", cfg.training.lora_layers])

    return command


def run_training(cfg: ProjectConfig, dry_run: bool = False) -> int:
    trainer_script = project_path(cfg.training.trainer_script)
    if not dry_run and not trainer_script.exists():
        raise FileNotFoundError(
            f"Trainer script not found: {trainer_script}\n"
            "Run `lora-pet fetch-trainer` first."
        )

    command = build_train_command(cfg)
    print(format_command(command))
    if dry_run:
        return 0
    return subprocess.call(command)


def format_command(command: list[str]) -> str:
    if sys.platform == "win32":
        return subprocess.list2cmdline(command)
    import shlex

    return shlex.join(command)


def _first_validation_prompt(cfg: ProjectConfig) -> str:
    if cfg.prompt.validation_prompts:
        return cfg.prompt.validation_prompts[0]
    return cfg.prompt.instance_prompt
