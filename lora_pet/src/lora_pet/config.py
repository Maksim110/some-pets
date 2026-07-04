from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


T = TypeVar("T")


@dataclass
class ProjectSection:
    name: str = "flux-lora-style-pet"
    seed: int = 42


@dataclass
class ModelSection:
    base_model: str = "black-forest-labs/FLUX.1-dev"
    torch_dtype: str = "bfloat16"


@dataclass
class DataSection:
    raw_dir: str = "data/raw/subject"
    processed_dir: str = "data/processed/subject"
    instance_dir: str = "data/processed/subject/instance"
    resolution: int = 512
    center_crop: bool = True
    min_size: int = 384
    duplicate_hamming_threshold: int = 4
    max_images: int = 24


@dataclass
class PromptSection:
    token: str = "mdvstyle"
    class_name: str = "object"
    instance_prompt: str = "a photo of mdvstyle object"
    validation_prompts: list[str] = field(default_factory=list)


@dataclass
class TrainingSection:
    output_dir: str = "outputs/flux_lora"
    trainer_script: str = "third_party/diffusers/examples/dreambooth/train_dreambooth_lora_flux.py"
    mixed_precision: str = "bf16"
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_train_steps: int = 300
    learning_rate: float = 1.0
    optimizer: str = "prodigy"
    lr_scheduler: str = "constant"
    lr_warmup_steps: int = 0
    rank: int = 8
    lora_alpha: int = 8
    guidance_scale: float = 1.0
    num_validation_images: int = 2
    validation_epochs: int = 50
    checkpointing_steps: int = 100
    gradient_checkpointing: bool = True
    cache_latents: bool = True
    allow_tf32: bool = True
    use_8bit_adam: bool = False
    report_to: str = "tensorboard"
    lora_layers: str | None = "attn.to_k,attn.to_q,attn.to_v,attn.to_out.0"


@dataclass
class InferenceSection:
    lora_dir: str = "outputs/flux_lora"
    adapter_name: str = "pet_lora"
    adapter_weight: float = 0.8
    output_dir: str = "outputs/samples"
    num_inference_steps: int = 28
    guidance_scale: float = 3.5
    height: int = 768
    width: int = 768
    prompts: list[str] = field(default_factory=list)


@dataclass
class ProjectConfig:
    project: ProjectSection = field(default_factory=ProjectSection)
    model: ModelSection = field(default_factory=ModelSection)
    data: DataSection = field(default_factory=DataSection)
    prompt: PromptSection = field(default_factory=PromptSection)
    training: TrainingSection = field(default_factory=TrainingSection)
    inference: InferenceSection = field(default_factory=InferenceSection)
    path: Path | None = None


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    cfg = ProjectConfig(
        project=_section(ProjectSection, raw.get("project", {})),
        model=_section(ModelSection, raw.get("model", {})),
        data=_section(DataSection, raw.get("data", {})),
        prompt=_section(PromptSection, raw.get("prompt", {})),
        training=_section(TrainingSection, raw.get("training", {})),
        inference=_section(InferenceSection, raw.get("inference", {})),
        path=config_path,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: ProjectConfig) -> None:
    if cfg.data.resolution <= 0:
        raise ValueError("data.resolution must be positive")
    if cfg.data.resolution % 8 != 0:
        raise ValueError("data.resolution should be divisible by 8 for latent diffusion models")
    if cfg.training.rank <= 0:
        raise ValueError("training.rank must be positive")
    if cfg.training.lora_alpha <= 0:
        raise ValueError("training.lora_alpha must be positive")
    if not cfg.prompt.token.strip():
        raise ValueError("prompt.token must not be empty")
    if cfg.prompt.token not in cfg.prompt.instance_prompt:
        raise ValueError("prompt.instance_prompt should contain prompt.token")


def project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _section(cls: type[T], data: dict[str, Any]) -> T:
    defaults = cls()
    names = {field.name for field in fields(cls)}
    unknown = sorted(set(data) - names)
    if unknown:
        joined = ", ".join(unknown)
        raise ValueError(f"Unknown config keys for {cls.__name__}: {joined}")

    values = {name: getattr(defaults, name) for name in names}
    values.update(data)
    return cls(**values)
