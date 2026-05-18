"""Phase 28 (GAN-leader scout) -- compare Qwen2.5-1.5B Krylov subspace to GPT-2 small on the same emotion corpus.

Before building a full cross-model GAN-leader configuration (translator T:
h_gpt2 -> h_qwen with adversarial discriminator), measure the prize: does
a 12x-larger model find a sharper / lower-dim emotion concept subspace
than GPT-2 small? If yes, the GAN-leader work is worth building. If no
(subspaces are similar), GAN-leader probably will not buy much.

Procedure:

1. Reuse Step 6b's 5-class emotion corpus (50 sentences x 5 classes = 100 total).
2. Harvest mid-layer hidden states from Qwen2.5-1.5B in bf16 on GPU.
3. Train the same predictive projection (32-d z, 64 hidden) on Qwen activations.
4. Compute gradient-Krylov SVD, project into top-3, save artefacts.
5. Compare directly to Step 6b's GPT-2 numbers.

If Qwen achieves substantially better separation (lower eff_rank, higher
top-3, visually cleaner clusters) the GAN-leader configuration with Qwen
as teacher is well-motivated. If not, this scout saves us from building
the full pipeline.
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

from nga.arch.predictive_projection import (  # noqa: E402
    PredictiveProjection,
    PredictiveProjectionConfig,
    train_predictive_projection,
)

# Reuse the corpus from Step 6b.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from phase27_step6b_emotion_krylov import CLASSES, CORPUS  # noqa: E402

SEED = 42
MODEL_ID = "Qwen/Qwen2.5-1.5B"
HARVEST_LAYER_FRAC = 0.5  # mid-layer as a fraction of total depth
PROJECTION_EPOCHS = 60
Z_DIM = 32
PROJ_HIDDEN = 64


class _QwenSubstrate:
    """Slim substrate wrapper for a generic causal LM (Qwen here).

    Mirrors _PretrainedSubstrate's interface enough to harvest sentence-level
    last-token hidden states. Loaded in bf16 to fit a 1.5B model on a 6 GB GPU.
    """

    def __init__(self, model_id: str = MODEL_ID, layer_frac: float = HARVEST_LAYER_FRAC,
                 device: str | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_id = model_id
        self._tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16
        )
        self._model.eval()
        self._model.to(self.device)
        for p in self._model.parameters():
            p.requires_grad_(False)

        self.hidden_size = int(self._model.config.hidden_size)
        self._n_layers = int(self._model.config.num_hidden_layers)
        self.harvest_layer = int(self._n_layers * layer_frac)
        if not (0 <= self.harvest_layer <= self._n_layers):
            raise ValueError(f"harvest_layer {self.harvest_layer} out of range")
        self.max_context = int(getattr(self._model.config, "max_position_embeddings", 2048))

    def harvest_sentence(self, text: str) -> np.ndarray:
        enc = self._tokenizer(text, return_tensors="pt", add_special_tokens=False,
                              truncation=True, max_length=self.max_context)
        input_ids = enc["input_ids"].to(self.device)
        with torch.no_grad():
            out = self._model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # hidden_states tuple length = n_layers + 1 (embedding output + each block).
        layer_out = out.hidden_states[self.harvest_layer][0]  # (seq_len, hidden_size)
        return layer_out[-1].float().detach().cpu().numpy().astype(np.float32)


def harvest_all(substrate: _QwenSubstrate, sentences: list[str]) -> np.ndarray:
    H = np.zeros((len(sentences), substrate.hidden_size), dtype=np.float32)
    for i, text in enumerate(sentences):
        H[i] = substrate.harvest_sentence(text)
    return H


def gradient_matrix(projection: PredictiveProjection, h: np.ndarray) -> np.ndarray:
    n, d = h.shape
    G = np.zeros((n, d), dtype=np.float32)
    projection.eval()
    for i in range(n):
        h_i = torch.from_numpy(h[i:i + 1]).requires_grad_(True)
        out = projection.forward(h_i)
        logits = out["next_state_logits"][0]
        top2 = torch.topk(logits, k=2, largest=True, sorted=True)
        margin = top2.values[0] - top2.values[1]
        margin.backward()
        G[i] = h_i.grad.detach().cpu().numpy()[0]
    return G


def krylov_stats(G: np.ndarray) -> dict:
    _U, sigmas, Vt = np.linalg.svd(G, full_matrices=False)
    variances = sigmas ** 2
    total = float(variances.sum())
    p = variances / max(total, 1e-30)
    nonzero = p > 0
    H_entropy = float(-(p[nonzero] * np.log(p[nonzero])).sum())
    return {
        "n_singular_values": int(sigmas.size),
        "total_variance": total,
        "effective_rank": float(np.exp(H_entropy)),
        "var_frac_top3": float(p[:3].sum()),
        "var_frac_top5": float(p[:5].sum()),
        "var_frac_top10": float(p[:10].sum()),
        "sigma_top_k": [float(s) for s in sigmas[:20]],
        "V_top3": Vt[:3].tolist(),
    }


def main() -> int:
    out_dir = RUNS_DIR / "phase28_gan_leader_scout"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    sentences: list[str] = []
    labels: list[int] = []
    for label, name in enumerate(CLASSES):
        for s in CORPUS[name]:
            sentences.append(s)
            labels.append(label)
    labels_arr = np.array(labels, dtype=np.int64)
    order = rng.permutation(len(sentences))
    sentences = [sentences[i] for i in order]
    labels_arr = labels_arr[order]

    print("=" * 100)
    print(f"Phase 28 GAN-leader scout: emotion Krylov on {MODEL_ID}")
    print(f"  classes      : {CLASSES}")
    print(f"  N sentences  : {len(sentences)} (20 per class)")
    print("=" * 100)

    t_start = time.perf_counter()
    substrate = _QwenSubstrate(model_id=MODEL_ID)
    print(f"  substrate ready : {substrate.model_id}")
    print(f"  hidden_size     : {substrate.hidden_size}")
    print(f"  layers          : {substrate._n_layers}")
    print(f"  harvest_layer   : block {substrate.harvest_layer} of {substrate._n_layers}")
    print(f"  device          : {substrate.device}")

    t_h = time.perf_counter()
    H = harvest_all(substrate, sentences)
    print(f"  harvest         : {H.shape}  ({time.perf_counter() - t_h:.1f}s)")

    cfg = PredictiveProjectionConfig(
        z_dim=Z_DIM,
        hidden_dim=PROJ_HIDDEN,
        n_states=len(CLASSES),
        entropy_weight=0.0,
        failure_weight=0.0,
    )
    projection = PredictiveProjection(input_dim=substrate.hidden_size, config=cfg)
    diag = train_predictive_projection(
        projection,
        H,
        next_states=labels_arr,
        entropy_targets=None,
        failure_targets=None,
        token_ids=None,
        epochs=PROJECTION_EPOCHS,
        seed=SEED,
    )
    print(f"  projection done : final_loss={diag.get('final_loss', float('nan')):.4f}  "
          f"acc={diag.get('next_state_acc', float('nan')):.3f}")

    G = gradient_matrix(projection, H)
    kry = krylov_stats(G)
    print(f"  Krylov SVD      : eff_rank(grad)={kry['effective_rank']:.2f}  "
          f"top-3={kry['var_frac_top3']:.3f}  top-5={kry['var_frac_top5']:.3f}  "
          f"top-10={kry['var_frac_top10']:.3f}")

    H_centered = H - H.mean(axis=0, keepdims=True)
    V_top3 = np.array(kry["V_top3"], dtype=np.float32)
    h_3d = H_centered @ V_top3.T
    np.savez_compressed(out_dir / "emotion_3d_data.npz",
                        h_3d=h_3d, labels=labels_arr, classes=np.array(CLASSES))

    print()
    print("Comparison to Step 6b (GPT-2 small, block 6, same emotion corpus):")
    gpt2_baseline = {
        "model": "gpt2 (124M, block 6 of 12)",
        "hidden_size": 768,
        "eff_rank_grad": 3.94,
        "top3": 0.816,
        "top5": 0.999,
    }
    print(f"  {'metric':>20} | {'GPT-2 (Step 6b)':>16} | {MODEL_ID:>22}")
    print(f"  {'-' * 20} | {'-' * 16} | {'-' * 22}")
    print(f"  {'hidden_size':>20} | {gpt2_baseline['hidden_size']:>16} | {substrate.hidden_size:>22}")
    print(f"  {'eff_rank(grad)':>20} | {gpt2_baseline['eff_rank_grad']:>16.2f} | {kry['effective_rank']:>22.2f}")
    print(f"  {'top-3 capture':>20} | {gpt2_baseline['top3']:>16.3f} | {kry['var_frac_top3']:>22.3f}")
    print(f"  {'top-5 capture':>20} | {gpt2_baseline['top5']:>16.3f} | {kry['var_frac_top5']:>22.3f}")

    kry_save = {k: v for k, v in kry.items() if k != "V_top3"}
    kry_save["V_top3"] = None
    summary = {
        "seed": SEED,
        "classes": CLASSES,
        "n_sentences": len(sentences),
        "model_id": substrate.model_id,
        "hidden_size": substrate.hidden_size,
        "n_layers": substrate._n_layers,
        "harvest_layer": substrate.harvest_layer,
        "kry": kry_save,
        "projection_diag": diag,
        "gpt2_baseline_step6b": gpt2_baseline,
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    (RUNS_DIR / "phase28_gan_leader_scout.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {RUNS_DIR / 'phase28_gan_leader_scout.json'}")
    print(f"  total wall-clock: {summary['total_wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
