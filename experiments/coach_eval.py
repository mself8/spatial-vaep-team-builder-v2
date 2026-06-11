"""Score the coach's actual XI with the frozen evaluator — cross-check of Table 3's 'Coach actual' row.

The main reproduction path is the s2_coach_vaep column that squadhan.train_e2e_vaep writes
to *_test_cv.csv; this script verifies the same numbers by recomputing them via an independent path (teambuilder_baseline.score_xi).

Usage:  COACH_EVAL_TAG=_gksel_sc_lc10_diff_cskip_vskip COACH_VALUE_SKIP=1 \
       python -m experiments.coach_eval
"""
import os
os.environ.setdefault("EDGE_SCALAR", "1")
os.environ.setdefault("OBJECTIVE", "vaep")
import numpy as np
import torch
from squadhan.train_e2e_vaep import (_build_yvaep_map, _build_gkids, _load_samples_fold,
                                     SEASONS, SEED, CHECKPOINTS_DIR)
from experiments.nb_helpers import load_model
from experiments.teambuilder_baseline import score_xi

TAG = os.environ.get("COACH_EVAL_TAG", "_gksel_sc_lc10_diff_cskip_vskip")
VALUE_SKIP = os.environ.get("COACH_VALUE_SKIP", "1") == "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
raw = _build_yvaep_map()
ymap = {(g, ih): v - raw[(g, 1 - ih)] for (g, ih), v in raw.items() if (g, 1 - ih) in raw}
gkids = _build_gkids()
out = []
for k in range(5):
    train_s, _, test_s = _load_samples_fold(SEASONS[k], SEED + k, ymap, gkids)
    ys = np.array([d._yv for d in train_s])
    mu, sd = float(ys.mean()), float(ys.std() + 1e-8)
    w = load_model(CHECKPOINTS_DIR / f"e2e_vaep_scalar{TAG}_stage2_fold{k}.pt",
                   gk_select=True, coord_skip=True, value_skip=VALUE_SKIP, device=device)
    vals = []
    for d in test_s:
        d = d.to(device)
        pid = d['our_squad'].player_ids
        gk = int(pid[0].item())                       # node0 = actual starting GK
        of = sorted(int(i) for i in pid[1:][d.our_starter_of_pool_idx.view(-1).long()])
        vals.append(score_xi(w, d, gk, of, mu, sd))
    out.append(float(np.mean(vals)))
    print(f"fold{k} {SEASONS[k]}: coach VAEP={out[-1]:+.3f}", flush=True)
a = np.array(out)
print(f"coach mean {a.mean():+.3f} ± {a.std(ddof=1):.3f}")
