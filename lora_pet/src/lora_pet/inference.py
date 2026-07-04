from __future__ import annotations

import json
from pathlib import Path

from .config import ProjectConfig, project_path


def run_inference(
    cfg: ProjectConfig,
    prompts: list[str] | None = None,
    num_images_per_prompt: int = 1,
    device: str | None = None,
) -> list[Path]:
    try:
        import torch
        from diffusers import FluxPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Inference requires training dependencies. Install them with "
            "`pip install -e .[train]` or `pip install -r requirements.txt`."
        ) from exc

    prompts = prompts or cfg.inference.prompts
    if not prompts:
        raise ValueError("No prompts provided for inference")

    output_dir = project_path(cfg.inference.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = _torch_dtype(torch, cfg.model.torch_dtype)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    pipe = FluxPipeline.from_pretrained(cfg.model.base_model, torch_dtype=torch_dtype)
    pipe.load_lora_weights(
        project_path(cfg.inference.lora_dir),
        adapter_name=cfg.inference.adapter_name,
    )
    pipe.set_adapters(cfg.inference.adapter_name, adapter_weights=cfg.inference.adapter_weight)
    pipe.to(target_device)

    generator = torch.Generator(device=target_device).manual_seed(cfg.project.seed)
    saved: list[Path] = []
    manifest: list[dict[str, object]] = []

    for prompt_idx, prompt in enumerate(prompts, start=1):
        result = pipe(
            prompt=prompt,
            num_images_per_prompt=num_images_per_prompt,
            num_inference_steps=cfg.inference.num_inference_steps,
            guidance_scale=cfg.inference.guidance_scale,
            height=cfg.inference.height,
            width=cfg.inference.width,
            generator=generator,
        )
        for image_idx, image in enumerate(result.images, start=1):
            image_path = output_dir / f"sample_{prompt_idx:02d}_{image_idx:02d}.png"
            image.save(image_path)
            saved.append(image_path)
            manifest.append(
                {
                    "path": str(image_path),
                    "prompt": prompt,
                    "adapter_weight": cfg.inference.adapter_weight,
                    "seed": cfg.project.seed,
                    "num_inference_steps": cfg.inference.num_inference_steps,
                    "guidance_scale": cfg.inference.guidance_scale,
                }
            )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return saved


def _torch_dtype(torch_module, name: str):
    normalized = name.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch_module.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch_module.float16
    if normalized in {"fp32", "float32"}:
        return torch_module.float32
    raise ValueError(f"Unsupported torch dtype: {name}")
