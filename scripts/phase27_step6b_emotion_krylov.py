"""Phase 27 Step 6b -- multi-class emotion Krylov (non-degenerate real-task test).

Step 6 (binary sentiment) hit eff_rank(grad) = 1.03 and top-3 = 0.998 -- both
acceptance bars passed but the projection memorized 100 sentences trivially
and the gradient collapsed to the single direction perpendicular to the
decision boundary. That is consistent with the framework's predictions for
a 2-class concept, but does not exercise the multi-dim subspace claim.

This step adds the non-trivial test: 5-class affective classification on a
balanced 100-sentence corpus (20 per class). With 5 classes the projection's
margin gradient cannot collapse to 1-d -- the decision boundary is at
minimum a 4-dim simplex in logit space, so the gradient subspace should
land at eff_rank in [3, 5] if the Step-4 dimensional-fingerprint claim
transfers cleanly.

Pre-registered acceptance:

  B1.  3 <= eff_rank(grad) <= 8           (non-degenerate AND not too high)
  B2.  top-3 capture in [0.50, 0.95]      (not 1.0; not below noise)
  B3.  cross-class separation visible in the 3-d Krylov projection
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
PROJECTION_EPOCHS = 60
Z_DIM = 32
PROJ_HIDDEN = 64

CLASSES = ["joy", "sadness", "anger", "fear", "surprise"]

CORPUS = {
    "joy": [
        "I cannot stop smiling, today has been absolutely wonderful.",
        "She laughed so hard tears ran down her cheeks.",
        "The kids danced around the room when they heard the news.",
        "What a delightful afternoon, the picnic was perfect.",
        "He grinned from ear to ear when he opened the gift.",
        "I am overjoyed that we finally got the offer.",
        "Her face lit up when she saw the puppy.",
        "We celebrated late into the night, everyone was in good spirits.",
        "The whole family burst out laughing at the joke.",
        "Such a happy ending, I am beaming with delight.",
        "He could not stop bouncing on his toes with excitement.",
        "She hugged her sister tightly, tears of joy in her eyes.",
        "What a beautiful day, the sun is shining and we are together.",
        "I felt a wave of happiness wash over me at the news.",
        "They sang along to the song at the top of their lungs.",
        "My heart is full, this is the best day of my life.",
        "The crowd erupted in cheers when the team scored.",
        "She twirled across the floor, giddy with happiness.",
        "I am thrilled beyond words, this is everything I hoped for.",
        "We shared a quiet smile, both of us were content.",
    ],
    "sadness": [
        "I have not stopped crying since I heard the news.",
        "She stared blankly at the floor, tears rolling silently down.",
        "He sat alone in the empty house, missing her terribly.",
        "What a heartbreaking ending, I felt hollow for hours.",
        "The funeral was unbearable, no one could speak.",
        "I am devastated, I do not know how to go on.",
        "Her shoulders shook as she sobbed into the pillow.",
        "We mourned the loss late into the night, no one slept.",
        "The whole room fell silent when she walked in alone.",
        "Such a tragic story, I felt grief sit on my chest.",
        "He could barely lift his head, weighed down with sorrow.",
        "She clutched his old sweater and wept for hours.",
        "What a bleak day, the rain matched my mood completely.",
        "I felt a wave of sadness wash over me without warning.",
        "They walked home in silence, both lost in their grief.",
        "My heart is broken, this is the worst week of my life.",
        "The crowd dispersed slowly, no one had the energy to talk.",
        "She turned her face to the wall, hiding her tears.",
        "I am heartbroken beyond words, nothing feels real anymore.",
        "We sat together quietly, both of us were grieving.",
    ],
    "anger": [
        "I am furious, I cannot believe they did this to us again.",
        "She slammed the door so hard the picture fell off the wall.",
        "He clenched his fists, struggling to keep his voice level.",
        "What an outrageous decision, I am beside myself with rage.",
        "The meeting ended in a screaming match, no one budged.",
        "I am livid, this is completely unacceptable behavior.",
        "Her face turned red as she shouted across the kitchen.",
        "We argued late into the night, both of us were seething.",
        "The whole team is fed up with management's broken promises.",
        "Such an infuriating result, I am ready to walk out.",
        "He stormed out without saying a single word to anyone.",
        "She gripped the steering wheel, knuckles white with anger.",
        "What an obnoxious display, I have never been so insulted.",
        "I felt a wave of fury rise up at the sheer hypocrisy.",
        "They yelled at each other across the parking lot for an hour.",
        "My patience is gone, I am about to lose it completely.",
        "The crowd booed angrily when the verdict was announced.",
        "She glared at him, jaw set, unwilling to give an inch.",
        "I am enraged beyond words, this betrayal will not stand.",
        "We stood toe to toe, both of us refusing to back down.",
    ],
    "fear": [
        "I am terrified, every shadow looks like someone is there.",
        "She froze in place, heart pounding, unable to move a muscle.",
        "He held his breath, listening for footsteps in the hallway.",
        "What a horrifying noise, I do not want to know what it was.",
        "The lights flickered and went out, I was suddenly alone.",
        "I am scared to death, please tell me everything will be okay.",
        "Her hands trembled as she dialed the emergency number.",
        "We crept through the dark warehouse, every sound made us jump.",
        "The whole family was afraid to step outside during the storm.",
        "Such a chilling encounter, my hair stood on end the whole time.",
        "He clutched the railing, paralyzed at the top of the stairs.",
        "She whispered for him to stay quiet, footsteps coming closer.",
        "What an unnerving silence, I could not shake the feeling of dread.",
        "I felt a wave of panic rise as the door handle turned slowly.",
        "They huddled together in the basement, listening for the sirens.",
        "My pulse is racing, I have never been this frightened before.",
        "The crowd grew quiet as the strange figure approached the stage.",
        "She backed away slowly, eyes locked on the dark shape in the corner.",
        "I am petrified beyond words, my legs will not carry me forward.",
        "We held onto each other, both of us afraid of what came next.",
    ],
    "surprise": [
        "I cannot believe it, I had absolutely no idea you were coming.",
        "She gasped audibly when she saw the room had been redecorated.",
        "He dropped his keys in shock when she opened the door.",
        "What an unexpected twist, none of us saw it coming at all.",
        "The whole room went silent when the door swung open.",
        "I am stunned, this is the last thing I expected to hear today.",
        "Her jaw actually dropped when the envelope was opened.",
        "We were all speechless, the announcement caught everyone off guard.",
        "The whole family was floored when grandpa walked through the door.",
        "Such an astonishing reveal, I had to sit down for a minute.",
        "He blinked twice, unsure if his eyes were playing tricks on him.",
        "She squealed in shock when the package was lifted off the truck.",
        "What an unbelievable coincidence, what are the odds of that.",
        "I felt a wave of astonishment ripple through the entire crowd.",
        "They exchanged looks, both of them too startled to speak.",
        "My mind is blown, I genuinely did not see that coming.",
        "The crowd gasped collectively when the magician vanished.",
        "She paused mid-sentence, processing what she had just heard.",
        "I am flabbergasted beyond words, I have so many questions.",
        "We sat in stunned silence for a full minute after the call ended.",
    ],
}

assert all(len(sents) == 20 for sents in CORPUS.values())
assert len(CORPUS) == 5


def harvest_sentence_hidden_states(substrate, sentences: list[str]) -> np.ndarray:
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
        layer_out = out.hidden_states[layer][0]
        H[i] = layer_out[-1].detach().cpu().numpy().astype(np.float32)
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
    out_dir = RUNS_DIR / "phase27_step6b_emotion"
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
    print("Phase 27 Step 6b -- 5-class emotion Krylov on GPT-2")
    print(f"  classes          : {CLASSES}")
    print(f"  N sentences      : {len(sentences)} (20 per class)")
    print(f"  harvest layer    : block {HARVEST_LAYER}")
    print(f"  acceptance       : 3 <= eff_rank(grad) <= 8 ; top-3 in [0.50, 0.95]")
    print("=" * 100)

    t_start = time.perf_counter()
    substrate = _PretrainedSubstrate(harvest_layer=HARVEST_LAYER)
    print(f"  substrate ready  : {substrate.model_id} on {substrate.device}")

    H = harvest_sentence_hidden_states(substrate, sentences)
    print(f"  harvest          : {H.shape}")

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
    print(f"  projection done  : final_loss={diag.get('final_loss', float('nan')):.4f}  "
          f"acc={diag.get('next_state_acc', float('nan')):.3f}")

    # Argmax label per sentence (the projection's class prediction).
    with torch.no_grad():
        logits = projection.forward(torch.from_numpy(H))["next_state_logits"]
        pred = logits.argmax(dim=-1).cpu().numpy()

    G = gradient_matrix(projection, H)
    kry = krylov_stats(G)
    print(f"  Krylov SVD       : eff_rank(grad)={kry['effective_rank']:.2f}  "
          f"top-3={kry['var_frac_top3']:.3f}  top-5={kry['var_frac_top5']:.3f}  "
          f"top-10={kry['var_frac_top10']:.3f}")

    H_centered = H - H.mean(axis=0, keepdims=True)
    V_top3 = np.array(kry["V_top3"], dtype=np.float32)
    h_3d = H_centered @ V_top3.T
    np.savez_compressed(out_dir / "emotion_3d_data.npz",
                        h_3d=h_3d, labels=labels_arr, pred=pred, classes=np.array(CLASSES))

    b1 = 3.0 <= kry["effective_rank"] <= 8.0
    b2 = 0.50 <= kry["var_frac_top3"] <= 0.95
    print()
    print("Acceptance:")
    print(f"  B1 eff_rank in [3, 8]  : {kry['effective_rank']:.2f}   {'PASS' if b1 else 'FAIL'}")
    print(f"  B2 top-3 in [0.5, 0.95]: {kry['var_frac_top3']:.3f}   {'PASS' if b2 else 'FAIL'}")
    print(f"  B3 visual separation  : check {out_dir.name}/emotion_3d.png")

    kry_save = {k: v for k, v in kry.items() if k != "V_top3"}
    kry_save["V_top3"] = None
    summary = {
        "seed": SEED,
        "classes": CLASSES,
        "n_sentences": len(sentences),
        "harvest_layer": HARVEST_LAYER,
        "projection_epochs": PROJECTION_EPOCHS,
        "model_id": substrate.model_id,
        "kry": kry_save,
        "projection_diag": diag,
        "acceptance": {
            "B1_eff_rank_in_[3,8]": b1,
            "B2_top3_in_[0.5,0.95]": b2,
            "B1_value": kry["effective_rank"],
            "B2_value": kry["var_frac_top3"],
        },
        "comparison_to_step4_grammars": {
            "step4_top3_range": [0.588, 0.888],
            "step4_eff_rank_range": [3.29, 10.89],
            "this_run_top3": kry["var_frac_top3"],
            "this_run_eff_rank": kry["effective_rank"],
        },
        "total_wall_clock_seconds": time.perf_counter() - t_start,
    }
    (RUNS_DIR / "phase27_step6b_emotion.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {RUNS_DIR / 'phase27_step6b_emotion.json'}")
    print(f"  total wall-clock: {summary['total_wall_clock_seconds']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
