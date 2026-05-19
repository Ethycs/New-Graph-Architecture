"""Phase 28 -- train a small SAE on Phase 27 Step 6b emotion-corpus activations.

Trains a `SparseAutoencoder` with 4x expansion ratio on GPT-2 small
block-6 activations harvested from the 5-class emotion corpus (Step 6b).
Saves the result to a numpy .npz checkpoint readable by
`PretrainedSAEAdapter.from_checkpoint`.

This is the first concrete artefact of Phase 28: a real SAE checkpoint
on disk that the labelled hypergraph can consume.

Outputs:
  * runs/phase28_sae/emotion_block6.npz -- the SAE checkpoint
  * runs/phase28_sae/training_log.json  -- final-epoch stats + config
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
os.environ["HF_HOME"] = str(Path.home() / "models" / "hf")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import torch  # noqa: E402

from nga.arch.sparse_autoencoder import (  # noqa: E402
    SparseAutoencoder,
    SparseAutoencoderConfig,
    train_sparse_autoencoder,
)
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402

# Reuse the emotion corpus from Step 6b.
from phase27_step6b_emotion_krylov import CLASSES, CORPUS  # noqa: E402

SEED = 42
HARVEST_LAYER = 6
EXPANSION_RATIO = 4  # n_features = 4 * d_in = 3072 for GPT-2 small
SPARSITY_COEF = 1e-3
EPOCHS = 200
BATCH_SIZE = 32
LR = 1e-3


def harvest_emotion_corpus(substrate: _PretrainedSubstrate) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Harvest last-token block-6 hidden states for the 100-sentence emotion corpus."""
    tok = substrate._tokenizer
    model = substrate._model
    device = substrate.device
    layer = substrate.harvest_layer
    sentences: list[str] = []
    labels: list[int] = []
    for label, name in enumerate(CLASSES):
        for s in CORPUS[name]:
            sentences.append(s)
            labels.append(label)
    H = np.zeros((len(sentences), substrate.hidden_size), dtype=np.float32)
    for i, text in enumerate(sentences):
        enc = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=substrate.max_context)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        H[i] = out.hidden_states[layer][0, -1].detach().cpu().numpy().astype(np.float32)
    return H, np.asarray(labels, dtype=np.int64), sentences


def main() -> int:
    out_dir = RUNS_DIR / "phase28_sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Phase 28 -- train SparseAutoencoder on emotion-corpus block-6 activations")
    print(f"  expansion_ratio  : {EXPANSION_RATIO}x")
    print(f"  sparsity_coef    : {SPARSITY_COEF}")
    print(f"  epochs           : {EPOCHS}")
    print(f"  seed             : {SEED}")
    print("=" * 100)

    t = time.perf_counter()
    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    print(f"  substrate ready  : {substrate.model_id} on {substrate.device}")

    H, labels, sentences = harvest_emotion_corpus(substrate)
    print(f"  harvest          : {H.shape}  ({(time.perf_counter() - t):.1f}s)")
    # Free the substrate weights from GPU to make room for SAE training.
    del substrate._model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d_in = int(H.shape[1])
    n_features = EXPANSION_RATIO * d_in
    print(f"  SAE              : d_in={d_in}, n_features={n_features}")

    torch.manual_seed(SEED)
    sae = SparseAutoencoder(SparseAutoencoderConfig(
        d_in=d_in, n_features=n_features, sparsity_coef=SPARSITY_COEF,
    ))
    t2 = time.perf_counter()
    final_stats = train_sparse_autoencoder(
        sae, H, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED, verbose=True,
    )
    train_time = time.perf_counter() - t2

    sae_path = out_dir / "emotion_block6.npz"
    sae.save_npz(sae_path)
    print(f"  saved SAE         : {sae_path}")

    # Also save the harvested activations + labels for downstream scripts
    # (avoid re-harvesting).
    np.savez_compressed(out_dir / "emotion_harvest.npz",
                        H=H, labels=labels, sentences=np.array(sentences))

    log = {
        "seed": SEED,
        "harvest_layer": HARVEST_LAYER,
        "expansion_ratio": EXPANSION_RATIO,
        "sparsity_coef": SPARSITY_COEF,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "d_in": d_in,
        "n_features": n_features,
        "n_samples": int(H.shape[0]),
        "final_loss_total": final_stats["loss_total"],
        "final_loss_recon": final_stats["loss_recon"],
        "final_loss_sparsity": final_stats["loss_sparsity"],
        "final_n_active_per_sample": final_stats["n_active_per_sample"],
        "training_wall_clock_seconds": train_time,
    }
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    print()
    print("Final stats:")
    for k, v in log.items():
        print(f"  {k:>32}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
