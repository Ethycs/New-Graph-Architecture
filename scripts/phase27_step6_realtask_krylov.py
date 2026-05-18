"""Phase 27 Step 6 -- real-concept Krylov: does the dimensional fingerprint transfer?

Tests whether the result from Phase 27 Step 4 (top-3 Krylov directions of
nabla margin capture 0.59-0.89 of gradient variance across 5 synthetic
grammars) generalizes to a natural-text concept-classification task.

Setup mirrors Step 4 but:
  * Substrate stays the same: frozen GPT-2 small, block-6 hidden states.
  * Task is binary sentiment classification on a hand-constructed balanced
    corpus (no FSM, no synthetic grammar).
  * Harvest = the last-non-pad hidden state per sentence (the natural
    "sentence representation" location).
  * Margin = top1 - top2 of a binary projection.

Pre-registered acceptance bars (set in advance, falsifiable):

  A1.  top-3 Krylov capture of grad-margin variance > 0.50
  A2.  effective rank of gradient matrix < 20
  A3.  3-d Krylov projection visually separates pos / neg
       (qualitative; checked from the saved PNG)

If A1 or A2 fails, the synthetic-grammar regime does not transfer cleanly
to natural-text concepts under this protocol and the Step-4 claim is
empirically narrower than its presentation.
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
from nga.exp.e30_pcg_extractor_pretrained import _PretrainedSubstrate  # noqa: E402

SEED = 42
HARVEST_LAYER = 6
PROJECTION_EPOCHS = 50
Z_DIM = 32
PROJ_HIDDEN = 64

POSITIVE = [
    "This movie was absolutely fantastic, I loved every minute.",
    "Best meal I have ever had at a restaurant, will definitely come back.",
    "The hotel staff were incredibly helpful and the room was beautiful.",
    "I am thrilled with this purchase, it exceeded all my expectations.",
    "What a wonderful performance, the actors were superb.",
    "The concert was electrifying, easily the best one I have been to.",
    "This book is a masterpiece, I could not put it down.",
    "Excellent service, friendly staff, delicious food, ten out of ten.",
    "The flight was smooth and the crew was very kind.",
    "I am so happy with how my hair turned out, the stylist is amazing.",
    "Phenomenal experience from start to finish, highly recommended.",
    "The product works perfectly and arrived earlier than promised.",
    "Such a charming little cafe, the coffee was outstanding.",
    "I had a wonderful time visiting this museum, very informative.",
    "Their customer support is the best I have ever encountered.",
    "Brilliant writing, gripping plot, unforgettable characters.",
    "This is hands down the most comfortable chair I have ever owned.",
    "The view from our balcony was breathtaking every morning.",
    "She delivered an incredible speech that moved everyone in the room.",
    "Loved the soundtrack, the cinematography, and the pacing.",
    "Five-star experience, everything was thoughtfully arranged.",
    "Fantastic instructor, made the class genuinely enjoyable.",
    "The garden is gorgeous in the spring, a real hidden gem.",
    "Their pastries are divine, especially the chocolate croissants.",
    "Quick delivery, great packaging, item is exactly as described.",
    "What an inspiring documentary, deeply researched and beautifully shot.",
    "The kids had a blast at the park, we will definitely return.",
    "An absolute joy to work with, professional and creative.",
    "The new update made the app so much smoother and more intuitive.",
    "Incredible value for the price, I am very impressed.",
    "Their wine list is exceptional and the sommelier knows her stuff.",
    "Such a relaxing weekend, exactly what we needed.",
    "The new album is brilliant, every track is a hit.",
    "Loved the show, witty dialogue and excellent ensemble cast.",
    "Beautiful weather, friendly locals, unforgettable trip.",
    "Top notch quality, this is well worth the investment.",
    "The therapist is wonderful, I feel so much better.",
    "A genuinely heartwarming story with a satisfying ending.",
    "This is the best coffee I have had in years.",
    "The interface is clean, the docs are clear, the API just works.",
    "She did an outstanding job organizing the event.",
    "I am over the moon with the results, thank you so much.",
    "The pool area is gorgeous and the cocktails are excellent.",
    "Their workshops are engaging and the materials are top quality.",
    "Such a thoughtful gift, my mom absolutely loved it.",
    "The performance was magical, a once-in-a-lifetime evening.",
    "Friendly neighborhood with great restaurants and parks.",
    "I cannot recommend this dentist enough, gentle and thorough.",
    "The hike was stunning, fall colors were at their peak.",
    "Truly a delightful little bookstore, I found three treasures.",
]

NEGATIVE = [
    "This movie was a complete waste of time, I want my two hours back.",
    "Worst meal I have ever paid for, the food was cold and bland.",
    "The hotel was filthy and the staff were rude when we complained.",
    "I am extremely disappointed with this purchase, broken on arrival.",
    "What a terrible performance, the actors clearly did not rehearse.",
    "The concert was a disaster, the sound system kept failing.",
    "This book is unreadable, I gave up after the first chapter.",
    "Horrible service, indifferent staff, mediocre food, would not return.",
    "The flight was a nightmare, three hours late and lost my luggage.",
    "I am furious with how my hair turned out, the stylist butchered it.",
    "Awful experience from start to finish, do not waste your money.",
    "The product is defective and customer service refuses to help.",
    "Such a depressing little cafe, the coffee tasted burnt.",
    "I had a miserable time at this museum, exhibits were neglected.",
    "Their customer support is the worst I have ever encountered.",
    "Painful writing, predictable plot, forgettable characters.",
    "This is hands down the most uncomfortable chair I have ever owned.",
    "The view from our balcony was a brick wall every morning.",
    "She delivered a tedious speech that put everyone to sleep.",
    "Hated the soundtrack, the cinematography, and the pacing.",
    "One-star experience, everything was poorly arranged.",
    "Awful instructor, made the class genuinely unpleasant.",
    "The garden is a dump in the spring, total disappointment.",
    "Their pastries are stale, especially the chocolate croissants.",
    "Slow delivery, damaged packaging, item is not as described.",
    "What a boring documentary, shallow research and amateur shots.",
    "The kids were miserable at the park, we will not return.",
    "An absolute pain to work with, unprofessional and uncreative.",
    "The new update made the app so much buggier and more confusing.",
    "Terrible value for the price, I am very disappointed.",
    "Their wine list is dismal and the sommelier was condescending.",
    "Such a stressful weekend, exactly what we did not need.",
    "The new album is dreadful, every track is a flop.",
    "Hated the show, dull dialogue and overacting throughout.",
    "Awful weather, hostile locals, forgettable trip.",
    "Cheap quality, this is not worth the money at all.",
    "The therapist is dreadful, I feel worse after sessions.",
    "A genuinely tedious story with a frustrating ending.",
    "This is the worst coffee I have had in years.",
    "The interface is cluttered, the docs are wrong, the API breaks constantly.",
    "She did a terrible job organizing the event.",
    "I am crushed with the results, total waste of money.",
    "The pool area is gross and the cocktails are watered down.",
    "Their workshops are boring and the materials are flimsy.",
    "Such a thoughtless gift, my mom was clearly offended.",
    "The performance was painful, a once-in-a-lifetime disappointment.",
    "Sketchy neighborhood with mediocre restaurants and overgrown parks.",
    "I cannot recommend this dentist enough to avoid, painful and careless.",
    "The hike was awful, the trail was muddy and unmarked.",
    "Truly a dismal little bookstore, I left empty-handed.",
]

assert len(POSITIVE) == 50
assert len(NEGATIVE) == 50


def harvest_sentence_hidden_states(substrate, sentences: list[str]) -> np.ndarray:
    """Return (N, D) where row i is the last-non-pad block-6 hidden state of sentence i."""
    tokenizer = substrate._tokenizer
    model = substrate._model
    layer = substrate.harvest_layer
    device = substrate.device
    H = np.zeros((len(sentences), substrate.hidden_size), dtype=np.float32)
    for i, text in enumerate(sentences):
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False, truncation=True,
                        max_length=substrate.max_context)
        input_ids = enc["input_ids"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
        # GPT-2 hidden_states tuple: index k for k in [1, n_layers] is block k.
        layer_out = out.hidden_states[layer][0]  # (seq_len, D)
        # Last token is the natural sentence representation for causal LMs.
        H[i] = layer_out[-1].detach().cpu().numpy().astype(np.float32)
    return H


def gradient_matrix(projection: PredictiveProjection, h: np.ndarray) -> np.ndarray:
    """Per-row gradient of margin w.r.t. h on CPU (matches Step 4 setup)."""
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
    out_dir = RUNS_DIR / "phase27_step6_realtask"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(SEED)
    sentences = POSITIVE + NEGATIVE
    labels = np.array([1] * len(POSITIVE) + [0] * len(NEGATIVE), dtype=np.int64)
    # Shuffle (deterministic) so training does not see pos-then-neg block order.
    order = rng.permutation(len(sentences))
    sentences = [sentences[i] for i in order]
    labels = labels[order]

    print("=" * 100)
    print("Phase 27 Step 6 -- real-concept Krylov on GPT-2 sentiment")
    print(f"  N sentences      : {len(sentences)} (50 pos + 50 neg)")
    print(f"  harvest layer    : block {HARVEST_LAYER}")
    print(f"  projection       : {Z_DIM}-d, {PROJ_HIDDEN}-hidden, {PROJECTION_EPOCHS} epochs")
    print(f"  acceptance bars  : top-3 > 0.50, eff_rank(grad) < 20")
    print("=" * 100)

    t_start = time.perf_counter()
    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    print(f"  substrate ready  : {substrate.model_id} on {substrate.device}")

    t_h = time.perf_counter()
    H = harvest_sentence_hidden_states(substrate, sentences)
    print(f"  harvest          : {H.shape}  ({time.perf_counter() - t_h:.1f}s)")

    # Train binary projection: h -> z -> 2-class logits.
    cfg = PredictiveProjectionConfig(
        z_dim=Z_DIM,
        hidden_dim=PROJ_HIDDEN,
        n_states=2,  # binary sentiment
        entropy_weight=0.0,
        failure_weight=0.0,
    )
    projection = PredictiveProjection(input_dim=substrate.hidden_size, config=cfg)
    t_p = time.perf_counter()
    diag = train_predictive_projection(
        projection,
        H,
        next_states=labels,
        entropy_targets=None,
        failure_targets=None,
        token_ids=None,
        epochs=PROJECTION_EPOCHS,
        seed=SEED,
    )
    print(f"  projection trained: final_loss={diag.get('final_loss', float('nan')):.4f}  "
          f"next_state_acc={diag.get('next_state_acc', float('nan')):.3f}  "
          f"({time.perf_counter() - t_p:.1f}s)")

    # Gradient matrix and Krylov SVD.
    t_g = time.perf_counter()
    G = gradient_matrix(projection, H)
    kry = krylov_stats(G)
    print(f"  gradient SVD done : eff_rank(grad)={kry['effective_rank']:.2f}  "
          f"top-3={kry['var_frac_top3']:.3f}  top-5={kry['var_frac_top5']:.3f}  "
          f"top-10={kry['var_frac_top10']:.3f}  ({time.perf_counter() - t_g:.1f}s)")

    # Project into top-3 Krylov subspace.
    H_centered = H - H.mean(axis=0, keepdims=True)
    V_top3 = np.array(kry["V_top3"], dtype=np.float32)
    h_3d = H_centered @ V_top3.T
    np.savez_compressed(out_dir / "sentiment_3d_data.npz", h_3d=h_3d, labels=labels)

    # Acceptance check.
    a1_pass = kry["var_frac_top3"] > 0.50
    a2_pass = kry["effective_rank"] < 20.0
    print()
    print("Acceptance:")
    print(f"  A1 top-3 > 0.50      : {kry['var_frac_top3']:.3f}   {'PASS' if a1_pass else 'FAIL'}")
    print(f"  A2 eff_rank < 20     : {kry['effective_rank']:.2f}   {'PASS' if a2_pass else 'FAIL'}")
    print(f"  A3 visual separation : check {out_dir.name}/krylov_3d.png")

    # Save summary.
    kry_save = {k: v for k, v in kry.items() if k != "V_top3"}
    kry_save["V_top3"] = None  # don't bloat the JSON
    summary = {
        "seed": SEED,
        "n_sentences": len(sentences),
        "harvest_layer": HARVEST_LAYER,
        "projection_epochs": PROJECTION_EPOCHS,
        "model_id": substrate.model_id,
        "kry": kry_save,
        "projection_diag": diag,
        "acceptance": {
            "A1_top3_above_0.50": a1_pass,
            "A2_eff_rank_below_20": a2_pass,
            "A1_value": kry["var_frac_top3"],
            "A2_value": kry["effective_rank"],
        },
        "comparison_to_step4_grammars": {
            "step4_top3_range": [0.588, 0.888],
            "step4_eff_rank_range": [3.29, 10.89],
        },
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    (RUNS_DIR / "phase27_step6_realtask.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {RUNS_DIR / 'phase27_step6_realtask.json'}")
    print(f"  total wall-clock: {summary['total_wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
