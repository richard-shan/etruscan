#!/usr/bin/env bash
set -euo pipefail

################################################################################
# Larth Etruscan NLP – Reproduction Wrapper
# -----------------------------------------
# This script re-runs the full training/evaluation pipeline documented in
# “Larth: dataset and machine translation for Etruscan”.  Run it inside the
# project root after activating the conda env defined in `environment.yml`.
# It assumes the host has a CUDA-capable GPU visible to JAX.
#
# The script performs the following steps:
#   1. Baseline sanity checks (random + dictionary models on the ETP split).
#   2. Prepare dedicated train configs for the transformer runs.
#   3. Train Larth on ETP-only data.
#   4. Train Larth on the combined ETP+CIEP corpus.
#   5. Evaluate the latest checkpoints for both models on their respective
#      held-out splits, reporting BLEU / chr-F / TER.
#
# All heavyweight computations happen inside the embedded Python snippets and
# the calls to `run_train.py`.  The script itself is idempotent; rerunning it
# overwrites the generated configs and metrics artefacts.
################################################################################

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RESULTS_DIR="${REPO_ROOT}/artifacts"
CONFIG_DIR="${RESULTS_DIR}/configs"
LOG_DIR="${RESULTS_DIR}/logs"
mkdir -p "${RESULTS_DIR}" "${CONFIG_DIR}" "${LOG_DIR}"

MODEL_CONFIG="${REPO_ROOT}/Translation/Larth/model.yml"

echo "[Step 0] Environment check"
python - <<'PY'
import importlib
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
missing = [pkg for pkg in required if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        f"Missing required packages: {missing}. "
        "Make sure `conda env create -f environment.yml` was executed."
    )
print("Python dependencies OK.")
PY

echo "[Step 1] Baseline models (random + dictionary)"
python - <<'PY'
import json
import numpy as np
from pathlib import Path

import Data
from Translation.random_models import RandomModel
from Translation.dictionary_model import DictionaryTranslation
from Translation.translation_utils import compute_metrics

OUTPUT_PATH = Path("artifacts") / "baseline_metrics_etp.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

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

etruscan, english = Data.load_translation_dataset(subset="etp")
train_et, train_en, test_et, test_en = split_data(
    etruscan, english, train_size=0.95, seed=0
)

tokenizer = Data.BlankspaceTokenizer()

# Random baselines (10 seeds, Gaussian length prior)
random_scores = []
for seed in range(10):
    model = RandomModel(length_dist="normal", seed=seed, name=f"random_seed_{seed}")
    model.train(train_en, english_tokenizer=tokenizer.tokenize)
    preds = model.predict(test_et)
    random_scores.append(compute_metrics(preds, test_en))

# Dictionary baselines (plain + suffix-aware tokeniser)
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
OUTPUT_PATH.write_text(json.dumps(summary, indent=2))
print(f"Saved baseline metrics to {OUTPUT_PATH}")
PY

echo "[Step 2] Generate training configs"
cat > "${CONFIG_DIR}/train_etruscan_etp.yml" <<'YAML'
batch_size: 32
lr: 0.002
warmup_steps: 250
weight_decay: 0.0001
workdir: artifacts/runs/larth_etp
label_smoothing: 0.1
restore_checkpoints: False
beam_size: 4
length_penalty: 0.6
epochs: 1000
eval_every_epochs: 50
checkpoint_every_epochs: 50

dataset_type: csv
subset: "etp"
data_path: ../../Data/Etruscan.csv
etruscan_vocab: ../../Data/ETP_POS.csv

source_model: ../../Data/all_small
target_model: ../../Data/all_small
alignment: same
cached: True
seed: 0
train_size: 0.95

source_lang: etruscan
target_lang: english

train: True
eval: True
mode: translation

name_augmentation_max_replacements: 0
unk_augmentation_prob: 0.0
unk_augmentation_len: 1
unk_augmentation_iterations: 0
YAML

cat > "${CONFIG_DIR}/train_etruscan_both.yml" <<'YAML'
batch_size: 32
lr: 0.002
warmup_steps: 250
weight_decay: 0.0001
workdir: artifacts/runs/larth_etp_ciep
label_smoothing: 0.1
restore_checkpoints: False
beam_size: 4
length_penalty: 0.6
epochs: 1000
eval_every_epochs: 50
checkpoint_every_epochs: 50

dataset_type: csv
subset: "both"
data_path: ../../Data/Etruscan.csv
etruscan_vocab: ../../Data/ETP_POS.csv

source_model: ../../Data/all_small
target_model: ../../Data/all_small
alignment: same
cached: True
seed: 0
train_size: 0.95

source_lang: etruscan
target_lang: english

train: True
eval: True
mode: translation

name_augmentation_max_replacements: 0
unk_augmentation_prob: 0.0
unk_augmentation_len: 1
unk_augmentation_iterations: 0
YAML

echo "[Step 3] Train Larth on ETP (logs -> artifacts/logs/larth_etp.log)"
python Translation/Larth/run_train.py \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${CONFIG_DIR}/train_etruscan_etp.yml" \
  > "${LOG_DIR}/larth_etp.log" 2>&1

echo "[Step 4] Train Larth on ETP+CIEP (logs -> artifacts/logs/larth_etp_ciep.log)"
python Translation/Larth/run_train.py \
  --model-config "${MODEL_CONFIG}" \
  --train-config "${CONFIG_DIR}/train_etruscan_both.yml" \
  > "${LOG_DIR}/larth_etp_ciep.log" 2>&1

echo "[Step 5] Evaluate latest checkpoints"
python - <<'PY'
import json
import numpy as np
from pathlib import Path

from Translation import translation_utils
from Translation.Larth import data_utils, inference, larth, train_utils

ARTIFACTS = Path("artifacts")

def latest_checkpoint(workdir: Path) -> Path:
    candidates = []
    for child in workdir.iterdir():
        if child.is_dir() and child.name.isdigit():
            candidates.append(child)
    if not candidates:
        raise FileNotFoundError(f"No checkpoint directories found under {workdir}")
    latest = max(candidates, key=lambda p: int(p.name))
    return latest / "default"

def evaluate(model_cfg_path: Path, train_cfg_path: Path, tag: str) -> None:
    model_cfg_dict = train_utils.parse_config(str(model_cfg_path))
    train_cfg_dict = train_utils.parse_config(str(train_cfg_path))

    model_cfg = larth.LarthTranslationConfig(**model_cfg_dict)
    train_cfg = train_utils.TrainConfig(**train_cfg_dict)

    # Reproduce dataset split
    rng = np.random.RandomState(train_cfg.seed)
    source_tok, target_tok = data_utils.load_tokenizers(train_cfg)
    source_texts, target_texts = data_utils.load_data(train_cfg)
    source_texts, target_texts = data_utils.remove_invalid(source_texts, target_texts)
    train_source, train_target, test_source, test_target = data_utils.split(
        source_texts, target_texts, train_cfg, rng
    )

    ckpt = latest_checkpoint(Path(train_cfg.workdir))
    params = inference.load_params(str(ckpt))

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
    out_path = ARTIFACTS / f"larth_metrics_{tag}.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[{tag}] BLEU={metrics['bleu']:.3f} chrF={metrics['chrf']:.3f} TER={metrics['ter']:.3f}")

evaluate(Path("Translation/Larth/model.yml"), Path("artifacts/configs/train_etruscan_etp.yml"), "etp")
evaluate(Path("Translation/Larth/model.yml"), Path("artifacts/configs/train_etruscan_both.yml"), "etp_ciep")
PY

echo "Pipeline complete. Artefacts available under ${RESULTS_DIR}"
