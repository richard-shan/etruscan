"""
Minimal baseline runner for environments without Bash/WSL.

This mirrors the baseline section of `run_full_pipeline.sh`: it trains the
random and dictionary translation baselines on the ETP split and stores the
metrics in `artifacts/baseline_metrics_etp.json`.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
translation_dir = REPO_ROOT / "Translation"
if str(translation_dir) not in sys.path:
    sys.path.insert(0, str(translation_dir))

import Data  # noqa: E402
from Translation.random_models import RandomModel  # noqa: E402
from Translation.dictionary_model import DictionaryTranslation  # noqa: E402
from Translation.translation_utils import compute_metrics  # noqa: E402


def split_data(source, target, train_size, seed):
    rng = np.random.RandomState(seed)
    indices = np.arange(len(source))
    rng.shuffle(indices)
    cut = int(round(len(indices) * train_size))
    train_idx = indices[:cut]
    test_idx = indices[cut:]
    train_source = [source[i] for i in train_idx]
    train_target = [target[i] for i in train_idx]
    test_source = [source[i] for i in test_idx]
    test_target = [target[i] for i in test_idx]
    return train_source, train_target, test_source, test_target


def main() -> None:
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)

    etruscan, english = Data.load_translation_dataset(subset="etp")
    train_et, train_en, test_et, test_en = split_data(
        etruscan, english, train_size=0.95, seed=0
    )

    tokenizer = Data.BlankspaceTokenizer()

    random_scores = []
    for seed in range(3):
        model = RandomModel(length_dist="normal", seed=seed, name=f"random_seed_{seed}")
        model.train(train_en, english_tokenizer=tokenizer.tokenize)
        preds = model.predict(test_et)
        random_scores.append(compute_metrics(preds, test_en))

    dict_plain = DictionaryTranslation(
        dictionary="Data/ETP_POS.csv",
        tokenizer=Data.BlankspaceTokenizer(),
        unk="",
    )
    plain_preds = [dict_plain.predict(x) for x in test_et]
    dict_plain_scores = compute_metrics(plain_preds, test_en)

    dict_suffix = DictionaryTranslation(
        dictionary="Data/ETP_POS.csv",
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
    out_path = artifacts / "baseline_metrics_etp.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved baseline metrics to {out_path}")


if __name__ == "__main__":
    main()
