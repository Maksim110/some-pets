# FLUX LoRA Pet

A small pet project for a junior RnD/ML application: a reproducible DreamBooth LoRA pipeline
for a FLUX image generation model.

The project demonstrates basic work with diffusion models, PEFT/LoRA, `diffusers`,
`accelerate`, data preparation, inference, and simple experiment reporting.

## Commands

- `prepare-data` cleans images, resizes them, and filters tiny or near-duplicate samples.
- `fetch-trainer` downloads the official FLUX LoRA trainer from Hugging Face Diffusers.
- `train` builds and runs the `accelerate` training command.
- `infer` generates images with the trained LoRA adapter.
- `evaluate-data` and `evaluate-generations` run lightweight sanity checks.
- `make-report` writes a short experiment card.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[train,dev]"
```

Put 10-24 subject or style images into `data/raw/subject`, then run:

```powershell
python -m lora_pet.cli prepare-data --force
python -m lora_pet.cli fetch-trainer
python -m lora_pet.cli train --dry-run
```

For real training, accept access to `black-forest-labs/FLUX.1-dev`, log in, and use a suitable GPU:

```powershell
hf auth login
python -m lora_pet.cli train
python -m lora_pet.cli infer
python -m lora_pet.cli make-report
```

Main settings are in `configs/flux_lora_pet.yaml`.
