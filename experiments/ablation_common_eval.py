"""Score the ablation variants' (notrf, nognn) recommended XIs with the frozen winner (Ours) evaluator.

Supplements Table 2 — verifies component contributions in selection quality, not TF accuracy (§4.3).
Each variant's own-evaluator Δ is not comparable, since a flatter evaluator inflates it (easier to exploit) →
as in Table 3, score every variant's XI with the single winner Stage-2 evaluator.

Each fold: load the variant's stage2 ckpt (with its own architecture flags) → for each test match,
deterministically select GK top-1 + OF top-10 → score with the winner evaluator's score_xi
(de-standardized with the winner's train mu/sd, in VAEP_DIFF units) + SelAcc (recall of the coach's 10 OF).
The 'ours' variant is a pipeline sanity check (should ≈ reproduce the existing s2_model_vaep).

Run (from the repository root): env MIN_ELIG_MINUTES=900 GK_SELECT=1 \
      python -m experiments.ablation_common_eval
Output: outputs/metrics/e2e_vaep_scalar_ablation_common_eval_test_cv.csv
"""
import os

os.environ.setdefault("EDGE_SCALAR", "1")
os.environ.setdefault("OBJECTIVE", "vaep")

import numpy as np
import pandas as pd
import torch

from squadhan.train_e2e_vaep import (
    _build_yvaep_map, _build_gkids, _load_samples_fold, _selected_of_ids,
    SEASONS, SEED, CHECKPOINTS_DIR, METRICS_DIR,
)
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP
from experiments.nb_helpers import load_model, NODE_DIM, HIDDEN_CHANNELS, NUM_LAYERS, NUM_HEADS, DROPOUT
from experiments.teambuilder_baseline import score_xi

WINNER_TAG = os.environ.get("WINNER_TAG", "_gksel_sc_lc10_diff_cskip")
WINNER_VSKIP = os.environ.get("WINNER_VALUE_SKIP", "0") == "1"   # for loading the vskip evaluator/variants
# variant -> (ckpt tag, model flags)
VARIANTS = {
    "ours":  (WINNER_TAG, dict(no_gnn=False, no_transformer=False)),
    "notrf": (WINNER_TAG + "_notrf", dict(no_gnn=False, no_transformer=True)),
    "nognn": (WINNER_TAG + "_nognn", dict(no_gnn=True, no_transformer=False)),
}
_SUF = "" if WINNER_TAG == "_gksel_sc_lc10_diff_cskip" else WINNER_TAG
OUT = METRICS_DIR / f"e2e_vaep_scalar_ablation_common_eval{_SUF}_test_cv.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_variant(tag, fold, **flags):
    ckpt = CHECKPOINTS_DIR / f"e2e_vaep_scalar{tag}_stage2_fold{fold}.pt"
    if not ckpt.exists():
        return None
    m = E2ELineupOptimizerVAEP(
        node_dim=NODE_DIM, edge_dim=1, hidden=HIDDEN_CHANNELS,
        n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
        gk_select=True, objective="vaep", coord_skip=True,
        value_skip=WINNER_VSKIP, **flags,
    ).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
    m.eval()
    return m


@torch.no_grad()
def pick_xi(model, data):
    """Deterministic selection by the variant's selector: GK top-1 + OF top-10 → (gk_id, [of_ids])."""
    our_emb, opp_emb = model.encoder(data)
    opp_ctx = opp_emb.mean(dim=0)
    pid = data["our_squad"].player_ids
    is_gk = data["our_squad"].is_gk.view(-1).bool()
    gk_ids = pid[is_gk]
    _, _, gk_idx = model.gk_selector(
        our_emb[is_gk], opp_ctx, k=1, training=False,
        elig=getattr(data["our_squad"], "gk_elig", None))
    gk_id = int(gk_ids[gk_idx.view(-1)[0]].item())
    of_ids, _, _ = _selected_of_ids(model, data)
    return gk_id, sorted(int(i) for i in of_ids)


def main():
    raw = _build_yvaep_map()
    ymap = {(g, ih): v - raw[(g, 1 - ih)] for (g, ih), v in raw.items() if (g, 1 - ih) in raw}
    gkids = _build_gkids()
    rows = []
    for k in range(5):
        ts, seed_k = SEASONS[k], SEED + k
        train_s, _, test_s = _load_samples_fold(ts, seed_k, ymap, gkids)
        ys = np.array([d._yv for d in train_s])
        mu, sd = float(ys.mean()), float(ys.std() + 1e-8)
        winner = load_model(CHECKPOINTS_DIR / f"e2e_vaep_scalar{WINNER_TAG}_stage2_fold{k}.pt",
                            gk_select=True, no_gnn=False, coord_skip=True,
                            value_skip=WINNER_VSKIP, device=device)
        for name, (tag, flags) in VARIANTS.items():
            vm = winner if name == "ours" else load_variant(tag, k, **flags)
            if vm is None:
                print(f"fold{k} {name}: ckpt missing — skip", flush=True)
                continue
            vals, sel = [], []
            for data in test_s:
                data = data.to(device)
                gk_id, of_ids = pick_xi(vm, data)
                vals.append(score_xi(winner, data, gk_id, of_ids, mu, sd))
                pid = data["our_squad"].player_ids
                coach_of = set(pid[1:][data.our_starter_of_pool_idx.view(-1).long()].cpu().numpy().tolist())
                sel.append(len(set(of_ids) & coach_of) / 10.0)
            rows.append({"variant": name, "fold": k, "test_season": ts,
                         "common_model_vaep": float(np.mean(vals)),
                         "selection_acc": float(np.mean(sel))})
            print(f"fold{k} {name}: common VAEP={rows[-1]['common_model_vaep']:+.3f} "
                  f"SelAcc={rows[-1]['selection_acc']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    for name in df.variant.unique():
        d = df[df.variant == name]
        print(f"{name:6s} common VAEP {d.common_model_vaep.mean():+.3f} ± {d.common_model_vaep.std(ddof=1):.3f}"
              f"  SelAcc {d.selection_acc.mean():.3f} ({len(d)} folds)")


if __name__ == "__main__":
    main()
