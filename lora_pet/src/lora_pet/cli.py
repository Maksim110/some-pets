from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .config import load_config, project_path
from .dataset import prepare_dataset
from .evaluate import evaluate_dataset, evaluate_generations
from .inference import run_inference
from .report import write_experiment_card
from .training import fetch_trainer, run_training


DEFAULT_CONFIG = "configs/flux_lora_pet.yaml"


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lora-pet",
        description="Compact FLUX LoRA pet-project pipeline.",
    )
    subparsers = parser.add_subparsers(required=True)

    doctor = subparsers.add_parser("doctor", help="Check local dependencies and hardware.")
    doctor.set_defaults(func=cmd_doctor)

    fetch = subparsers.add_parser("fetch-trainer", help="Download the official Diffusers FLUX LoRA trainer.")
    fetch.add_argument("--ref", default="main", help="Git ref in huggingface/diffusers.")
    fetch.add_argument("--target-root", default="third_party/diffusers")
    fetch.set_defaults(func=cmd_fetch_trainer)

    prepare = subparsers.add_parser("prepare-data", help="Prepare subject images for DreamBooth LoRA.")
    add_config_arg(prepare)
    prepare.add_argument("--source-dir", default=None, help="Directory with raw subject images.")
    prepare.add_argument("--urls-file", default=None, help="Optional text file with image URLs to download.")
    prepare.add_argument("--caption", default=None, help="Caption stored in metadata.jsonl.")
    prepare.add_argument("--force", action="store_true", help="Remove previous processed images first.")
    prepare.set_defaults(func=cmd_prepare_data)

    train = subparsers.add_parser("train", help="Run or print the FLUX LoRA training command.")
    add_config_arg(train)
    train.add_argument("--dry-run", action="store_true", help="Print command without launching training.")
    train.set_defaults(func=cmd_train)

    infer = subparsers.add_parser("infer", help="Generate images with the trained LoRA adapter.")
    add_config_arg(infer)
    infer.add_argument("--prompt", action="append", default=None, help="Prompt. Can be passed multiple times.")
    infer.add_argument("--num-images-per-prompt", type=int, default=1)
    infer.add_argument("--device", default=None, help="cuda, cpu, or any torch device string.")
    infer.set_defaults(func=cmd_infer)

    eval_data = subparsers.add_parser("evaluate-data", help="Compute lightweight dataset quality checks.")
    add_config_arg(eval_data)
    eval_data.set_defaults(func=cmd_evaluate_data)

    eval_gen = subparsers.add_parser("evaluate-generations", help="Compute lightweight checks for generated images.")
    add_config_arg(eval_gen)
    eval_gen.add_argument("--images-dir", default=None)
    eval_gen.set_defaults(func=cmd_evaluate_generations)

    card = subparsers.add_parser("make-report", help="Write a compact experiment card.")
    add_config_arg(card)
    card.add_argument("--output", default="reports/experiment_card.md")
    card.set_defaults(func=cmd_make_report)

    return parser


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to YAML config.")


def cmd_doctor(_args: argparse.Namespace) -> None:
    modules = [
        "PIL",
        "yaml",
        "numpy",
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "peft",
        "safetensors",
    ]
    status = {module: importlib.util.find_spec(module) is not None for module in modules}

    cuda = None
    if status["torch"]:
        import torch

        cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    print(json.dumps({"modules": status, "cuda": cuda}, indent=2))


def cmd_fetch_trainer(args: argparse.Namespace) -> None:
    downloaded = fetch_trainer(ref=args.ref, target_root=args.target_root)
    for path in downloaded:
        print(path)


def cmd_prepare_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = prepare_dataset(
        cfg,
        source_dir=args.source_dir,
        urls_file=args.urls_file,
        caption=args.caption,
        force=args.force,
    )
    print(f"accepted={len(report.accepted)} skipped={report.skipped}")
    print(project_path(cfg.data.processed_dir) / "dataset_report.md")


def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    exit_code = run_training(cfg, dry_run=args.dry_run)
    if exit_code != 0:
        raise SystemExit(exit_code)


def cmd_infer(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    saved = run_inference(
        cfg,
        prompts=args.prompt,
        num_images_per_prompt=args.num_images_per_prompt,
        device=args.device,
    )
    for path in saved:
        print(path)


def cmd_evaluate_data(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = evaluate_dataset(cfg)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_evaluate_generations(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    report = evaluate_generations(cfg, images_dir=args.images_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_make_report(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    target = write_experiment_card(cfg, output_path=Path(args.output))
    print(target)

if __name__ == "__main__":
    main()
