"""
End-to-end pipeline driver implemented in Python (no shell dependencies).

Usage:
    python scripts/run_pipeline.py

This script validates the environment, runs the baselines, launches two Larth
training runs (ETP and ETP+CIEP), and evaluates the resulting checkpoints.
Artefacts are written under the `artifacts/` directory.
"""

from __future__ import annotations

import argparse
from importlib import util
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


REPO_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
CONFIG_DIR = ARTIFACTS_DIR / "configs"
LOG_DIR = ARTIFACTS_DIR / "logs"
RUNS_DIR = ARTIFACTS_DIR / "runs"


def ensure_paths() -> None:
    for path in (ARTIFACTS_DIR, CONFIG_DIR, LOG_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def ensure_dependencies() -> None:
    required = [
        "numpy",
        "pandas",
        "pyarrow",
        "sentencepiece",
        "absl",
        "jax",
        "flax",
        "optax",
        "orbax.checkpoint",
        "sacrebleu",
    ]
    missing: List[str] = []
    for pkg in required:
        if util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        raise SystemExit(
            "Missing required Python packages: "
            f"{', '.join(missing)}\n"
            "Install the dependencies manually (pip install ...) before rerunning."
        )


def run_baselines(seed_runs: int = 10) -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "Translation"))

    import numpy as np

    import Data
    from Translation.random_models import RandomModel
    from Translation.dictionary_model import DictionaryTranslation
    from Translation.translation_utils import compute_metrics

    def split_data(source, target, train_size, seed):
        rng = np.random.RandomState(seed)
        idx = np.arange(len(source))
        rng.shuffle(idx)
        cut = int(round(len(idx) * train_size))
        train_idx = idx[:cut]
        test_idx = idx[cut:]
        train_source = [source[i] for i in train_idx]
        train_target = [target[i] for i in train_idx]
        test_source = [source[i] for i in test_idx]
        test_target = [target[i] for i in test_idx]
        return train_source, train_target, test_source, test_target

    etruscan, english = Data.load_translation_dataset(subset="etp")
    train_et, train_en, test_et, test_en = split_data(
        etruscan, english, train_size=0.95, seed=0
    )

    tokenizer = Data.BlankspaceTokenizer()

    random_scores = []
    for seed in range(seed_runs):
        model = RandomModel(length_dist="normal", seed=seed, name=f"random_{seed}")
        model.train(train_en, english_tokenizer=tokenizer.tokenize)
        preds = model.predict(test_et)
        random_scores.append(compute_metrics(preds, test_en))

    dict_plain = DictionaryTranslation(
        dictionary=str(REPO_ROOT / "Data" / "ETP_POS.csv"),
        tokenizer=Data.BlankspaceTokenizer(),
        unk="",
    )
    plain_preds = [dict_plain.predict(x) for x in test_et]
    dict_plain_scores = compute_metrics(plain_preds, test_en)

    dict_suffix = DictionaryTranslation(
        dictionary=str(REPO_ROOT / "Data" / "ETP_POS.csv"),
        tokenizer=Data.SuffixTokenizer(),
        unk="",
    )
    suffix_preds = [dict_suffix.predict(x) for x in test_et]
    dict_suffix_scores = compute_metrics(suffix_preds, test_en)

    summary = {
        "dataset": "ETP",
        "train_size": 0.95,
        "random": random_scores,
        "dictionary_plain": dict_plain_scores,
        "dictionary_suffix": dict_suffix_scores,
    }
    out_path = ARTIFACTS_DIR / "baseline_metrics_etp.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"[Baselines] Metrics saved to {out_path}")
    return out_path


def write_config(run_name: str, subset: str) -> Path:
    config = {
        "batch_size": 32,
        "lr": 0.002,
        "warmup_steps": 250,
        "weight_decay": 0.0001,
        "workdir": str((RUNS_DIR / run_name).resolve()),
        "label_smoothing": 0.1,
        "restore_checkpoints": False,
        "beam_size": 4,
        "length_penalty": 0.6,
        "epochs": 1000,
        "eval_every_epochs": 50,
        "checkpoint_every_epochs": 50,
        "dataset_type": "csv",
        "subset": subset,
        "data_path": str((REPO_ROOT / "Data" / "Etruscan.csv").resolve()),
        "etruscan_vocab": str((REPO_ROOT / "Data" / "ETP_POS.csv").resolve()),
        "source_model": str((REPO_ROOT / "Data" / "all_small").resolve()),
        "target_model": str((REPO_ROOT / "Data" / "all_small").resolve()),
        "alignment": "same",
        "cached": True,
        "seed": 0,
        "train_size": 0.95,
        "source_lang": "etruscan",
        "target_lang": "english",
        "train": True,
        "eval": True,
        "mode": "translation",
        "name_augmentation_max_replacements": 0,
        "unk_augmentation_prob": 0.0,
        "unk_augmentation_len": 1,
        "unk_augmentation_iterations": 0,
    }
    out_path = CONFIG_DIR / f"train_{run_name}.json"
    out_path.write_text(json.dumps(config, indent=2))
    return out_path


def run_training(model_config: Path, train_config: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "Translation" / "Larth" / "run_train.py"),
        "--model-config",
        str(model_config),
        "--train-config",
        str(train_config),
    ]
    print(f"[Train] Running {' '.join(cmd)}")
    with log_path.open("w", encoding="utf-8") as log_f:
        subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, check=True)
    print(f"[Train] Finished. Log stored in {log_path}")


def latest_checkpoint(workdir: Path) -> Path:
    candidates = [p for p in workdir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found in {workdir}")
    latest = max(candidates, key=lambda p: int(p.name))
    return latest / "default"


def evaluate(model_config: Path, train_config: Path, tag: str) -> Path:
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "Translation"))

    import numpy as np

    from Translation import translation_utils
    from Translation.Larth import data_utils, inference, larth, train_utils

    model_cfg_dict = train_utils.parse_config(str(model_config))
    train_cfg_dict = train_utils.parse_config(str(train_config))

    model_cfg = larth.LarthTranslationConfig(**model_cfg_dict)
    train_cfg = train_utils.TrainConfig(**train_cfg_dict)

    rng = np.random.RandomState(train_cfg.seed)
    source_tok, target_tok = data_utils.load_tokenizers(train_cfg)

    source_texts, target_texts = data_utils.load_data(train_cfg)
    source_texts, target_texts = data_utils.remove_invalid(source_texts, target_texts)

    (
        train_source,
        train_target,
        test_source,
        test_target,
    ) = data_utils.split(source_texts, target_texts, train_cfg, rng)

    ckpt_path = latest_checkpoint(Path(train_cfg.workdir))
    params = inference.load_params(str(ckpt_path))

    _, predictions = inference.translate(
        test_source,
        params,
        source_tok,
        target_tok,
        batch_size=train_cfg.batch_size,
        beam_size=train_cfg.beam_size,
        max_len=model_cfg.max_len,
        model_config=model_cfg,
        clean_cache=True,
    )

    metrics = translation_utils.compute_metrics(predictions, test_target)
    out_path = ARTIFACTS_DIR / f"larth_metrics_{tag}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"[Eval:{tag}] BLEU={metrics['bleu']:.3f} "
        f"chrF={metrics['chrf']:.3f} TER={metrics['ter']:.3f}"
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full Larth pipeline.")
    parser.add_argument(
        "--skip-baselines", action="store_true", help="Skip baseline evaluation."
    )
    parser.add_argument(
        "--skip-training", action="store_true", help="Skip transformer training runs."
    )
    parser.add_argument(
        "--skip-eval", action="store_true", help="Skip final evaluation after training."
    )
    args = parser.parse_args()

    ensure_paths()
    ensure_dependencies()

    model_config = REPO_ROOT / "Translation" / "Larth" / "model.yml"

    if not args.skip_baselines:
        run_baselines()

    etp_config = write_config("larth_etp", "etp")
    both_config = write_config("larth_etp_ciep", "both")

    if not args.skip_training:
        run_training(
            model_config=model_config,
            train_config=etp_config,
            log_path=LOG_DIR / "larth_etp.log",
        )
        run_training(
            model_config=model_config,
            train_config=both_config,
            log_path=LOG_DIR / "larth_etp_ciep.log",
        )

    if not args.skip_eval:
        evaluate(model_config, etp_config, tag="etp")
        evaluate(model_config, both_config, tag="etp_ciep")

    print("Pipeline complete. Check the artifacts/ directory for results.")


if __name__ == "__main__":
    main()
