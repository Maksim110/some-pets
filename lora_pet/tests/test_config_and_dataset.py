from pathlib import Path

import yaml
from PIL import Image

from lora_pet.config import load_config
from lora_pet.dataset import hamming_distance, prepare_dataset


def test_prepare_dataset_filters_duplicates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw_dir = Path("data/raw/subject")
    raw_dir.mkdir(parents=True)
    Image.new("RGB", (512, 512), (240, 30, 30)).save(raw_dir / "one.jpg")
    Image.new("RGB", (512, 512), (240, 30, 30)).save(raw_dir / "duplicate.jpg")
    Image.new("RGB", (128, 128), (30, 240, 30)).save(raw_dir / "small.jpg")

    config_path = Path("config.yaml")
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "raw_dir": "data/raw/subject",
                    "processed_dir": "data/processed/subject",
                    "instance_dir": "data/processed/subject/instance",
                    "resolution": 512,
                    "min_size": 384,
                },
                "prompt": {
                    "token": "mdvstyle",
                    "instance_prompt": "a photo of mdvstyle object",
                },
            }
        ),
        encoding="utf-8",
    )

    cfg = load_config(config_path)
    report = prepare_dataset(cfg, force=True)

    assert len(report.accepted) == 1
    assert report.skipped["duplicate"] == 1
    assert report.skipped["too_small"] == 1
    assert Path("data/processed/subject/metadata.jsonl").exists()


def test_hamming_distance():
    assert hamming_distance("1010", "0011") == 2
