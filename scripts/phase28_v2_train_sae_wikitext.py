"""Phase 28 v2 -- train a larger SAE on WikiText-2 GPT-2 block-6 activations.

Phase 28 v1 trained on 100 hand-constructed emotion sentences. The resulting
SAE was useful as plumbing but the 39-of-3072 labelling rate reflected
the toy training corpus. v2 trains on WikiText-2's natural-language
distribution -- ~2M tokens -- and produces an SAE with much broader
feature coverage. The same `PretrainedSAEAdapter` interface consumes
both; the only thing that changes is the checkpoint quality.

Procedure:

  1. Load WikiText-2-raw train split from
     ``~/models/hf/hub/datasets--wikitext/snapshots/.../wikitext-2-raw-v1/``.
  2. Tokenize a sample of ~5K passages with GPT-2's BPE; harvest
     block-6 hidden states at random non-special token positions.
     Aim for ~20K-50K position samples (~15-40x the v1 corpus).
  3. Train an SAE with 5.3x expansion (4096 features) for more epochs.
  4. Save to ``runs/phase28_v2_sae/wikitext_block6.npz``.

Outputs:
  * runs/phase28_v2_sae/wikitext_block6.npz  -- the SAE checkpoint
  * runs/phase28_v2_sae/training_log.json    -- final stats + config
  * runs/phase28_v2_sae/wikitext_harvest.npz -- the harvested
    activations, for downstream labelling without re-harvest.
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

import torch  # noqa: E402

from nga.arch.sparse_autoencoder import (  # noqa: E402
    SparseAutoencoder,
    SparseAutoencoderConfig,
    train_sparse_autoencoder,
)
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402

SEED = 42
HARVEST_LAYER = 6
EXPANSION_RATIO = 5.3  # n_features = round(5.3 * 768) = 4070, round to 4096
N_FEATURES = 4096
SPARSITY_COEF = 1e-3
EPOCHS = 100  # fewer per-epoch passes since the corpus is 100x larger
BATCH_SIZE = 256
LR = 1e-3
TARGET_N_ACTIVATIONS = 24000  # ~24K position samples target
PASSAGES_TO_HARVEST = 4000  # number of WikiText lines/passages to sample
MIN_PASSAGE_LEN = 32  # skip short / empty lines


def find_wikitext_train_parquet() -> Path:
    """Locate the locally-cached wikitext-2-raw train parquet."""
    hub = Path.home() / "models" / "hf" / "hub" / "datasets--wikitext"
    for snap in (hub / "snapshots").iterdir():
        candidate = snap / "wikitext-2-raw-v1" / "train-00000-of-00001.parquet"
        if candidate.exists():
            return candidate
    raise FileNotFoundError("wikitext-2-raw train parquet not found under ~/models/hf/hub")


def load_wikitext_passages(parquet_path: Path, max_passages: int, seed: int) -> list[str]:
    """Load wikitext train passages, skipping empty/section-header lines."""
    import pyarrow.parquet as pq
    table = pq.read_table(parquet_path)
    texts = table.column("text").to_pylist()
    # Filter to substantive lines (skip empty + heading-style "= ... =" lines).
    passages = [
        t.strip()
        for t in texts
        if len(t.strip()) >= MIN_PASSAGE_LEN and not t.strip().startswith("=")
    ]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(passages))[:max_passages]
    return [passages[int(i)] for i in idx]


def harvest_wikitext_activations(
    substrate: _PretrainedSubstrate, passages: list[str], target_n: int,
) -> np.ndarray:
    """Harvest GPT-2 block-N hidden states at random non-special token positions.

    For each passage: tokenize, run forward, sample up to (target_n // n_passages)
    random token positions, collect their hidden states. Stop once `target_n`
    is reached.
    """
    model = substrate._model
    tok = substrate._tokenizer
    device = substrate.device
    layer = substrate.harvest_layer
    H_list: list[np.ndarray] = []
    rng = np.random.default_rng(SEED)
    per_passage_cap = max(1, target_n // len(passages))
    t = time.perf_counter()
    n_collected = 0
    n_passages_done = 0
    for text in passages:
        if n_collected >= target_n:
            break
        enc = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=substrate.max_context)
        input_ids = enc["input_ids"].to(device)
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        layer_out = out.hidden_states[layer][0]  # (seq_len, hidden_size)
        seq_len = layer_out.shape[0]
        # Random positions in this passage.
        n_take = min(per_passage_cap, seq_len)
        positions = rng.choice(seq_len, size=n_take, replace=False)
        for p in positions:
            H_list.append(layer_out[int(p)].detach().cpu().numpy().astype(np.float32))
            n_collected += 1
            if n_collected >= target_n:
                break
        n_passages_done += 1
        if n_passages_done % 200 == 0:
            elapsed = time.perf_counter() - t
            print(f"    harvested {n_collected:>6} positions from {n_passages_done:>4} passages "
                  f"({elapsed:.1f}s elapsed)")
    H = np.stack(H_list, axis=0)
    return H


def main() -> int:
    out_dir = RUNS_DIR / "phase28_v2_sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("Phase 28 v2 -- train SAE on WikiText-2 block-6 activations")
    print(f"  target_n_activations : {TARGET_N_ACTIVATIONS}")
    print(f"  n_features           : {N_FEATURES} ({N_FEATURES / 768:.1f}x expansion)")
    print(f"  sparsity_coef        : {SPARSITY_COEF}")
    print(f"  epochs               : {EPOCHS}")
    print(f"  batch_size           : {BATCH_SIZE}")
    print(f"  seed                 : {SEED}")
    print("=" * 100)

    parquet = find_wikitext_train_parquet()
    print(f"  loading wikitext     : {parquet}")
    passages = load_wikitext_passages(parquet, PASSAGES_TO_HARVEST, seed=SEED)
    print(f"  passages loaded      : {len(passages)}")
    print(f"  example passage      : {passages[0][:120]}...")

    t = time.perf_counter()
    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    print(f"  substrate ready      : {substrate.model_id} on {substrate.device}  "
          f"({time.perf_counter() - t:.1f}s)")

    t2 = time.perf_counter()
    H = harvest_wikitext_activations(substrate, passages, TARGET_N_ACTIVATIONS)
    harvest_time = time.perf_counter() - t2
    print(f"  harvest              : {H.shape}  ({harvest_time:.1f}s)")

    # Free substrate weights from GPU to make room for SAE training.
    del substrate._model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    d_in = int(H.shape[1])
    print(f"  SAE config           : d_in={d_in}, n_features={N_FEATURES}")

    torch.manual_seed(SEED)
    sae = SparseAutoencoder(SparseAutoencoderConfig(
        d_in=d_in, n_features=N_FEATURES, sparsity_coef=SPARSITY_COEF,
    ))
    if torch.cuda.is_available():
        sae = sae.cuda()
        print(f"  SAE on device       : cuda (params: {sum(p.numel() for p in sae.parameters()):,})")
    else:
        print(f"  SAE on device       : cpu (params: {sum(p.numel() for p in sae.parameters()):,})")
    t3 = time.perf_counter()
    final_stats = train_sparse_autoencoder(
        sae, H, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, seed=SEED, verbose=True,
    )
    train_time = time.perf_counter() - t3

    sae_path = out_dir / "wikitext_block6.npz"
    sae.save_npz(sae_path)
    print(f"  saved SAE            : {sae_path}")

    # Save the harvest too -- useful for re-labelling without re-harvest.
    np.savez_compressed(out_dir / "wikitext_harvest.npz",
                        H=H[:min(H.shape[0], 5000)].astype(np.float32))  # cap size on disk

    log = {
        "seed": SEED,
        "harvest_layer": HARVEST_LAYER,
        "n_features": N_FEATURES,
        "expansion_ratio": N_FEATURES / d_in,
        "sparsity_coef": SPARSITY_COEF,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "d_in": d_in,
        "n_train_activations": int(H.shape[0]),
        "n_passages_used": len(passages),
        "harvest_time_seconds": harvest_time,
        "training_time_seconds": train_time,
        "final_loss_total": final_stats["loss_total"],
        "final_loss_recon": final_stats["loss_recon"],
        "final_loss_sparsity": final_stats["loss_sparsity"],
        "final_n_active_per_sample": final_stats["n_active_per_sample"],
    }
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))

    print()
    print("Final stats:")
    for k, v in log.items():
        print(f"  {k:>32}: {v}")
    print()
    print(f"v1 vs v2 quick comparison:")
    v1_log = json.loads((RUNS_DIR / "phase28_sae" / "training_log.json").read_text())
    print(f"  {'metric':>30} | {'v1':>12} | {'v2':>12}")
    print(f"  {'-' * 30} | {'-' * 12} | {'-' * 12}")
    print(f"  {'n_features':>30} | {v1_log['n_features']:>12} | {N_FEATURES:>12}")
    print(f"  {'n_train_activations':>30} | {v1_log['n_samples']:>12} | {H.shape[0]:>12}")
    print(f"  {'final_recon':>30} | {v1_log['final_loss_recon']:>12.5f} | {final_stats['loss_recon']:>12.5f}")
    print(f"  {'final_n_active_per_sample':>30} | "
          f"{v1_log['final_n_active_per_sample']:>12.1f} | "
          f"{final_stats['n_active_per_sample']:>12.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
