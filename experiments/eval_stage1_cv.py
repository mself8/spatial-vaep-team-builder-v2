"""Generate the 5-fold test metrics → test_cv.csv for stage-1-only (ablation) runs.

Computes only the s1 metrics, with the same definitions as train_e2e_vaep's evaluate_test,
and saves them to outputs/metrics/e2e_vaep_scalar{TAG}_test_cv.csv (s1_* columns).
The w/o GNN and w/o Transformer rows of Table 2 are stage1-only runs for which the trainer
writes no test_cv, so this script fills that gap.

Usage:
  python -m experiments.eval_stage1_cv --ckpt-tag _gksel_sc_lc10_diff_cskip_vskip_notrf \
      --no-transformer --coord-skip --value-skip
"""
import argparse
import os

import pandas as pd
import torch

os.environ.setdefault("EDGE_SCALAR", "1")
os.environ.setdefault("GK_SELECT", "1")
os.environ.setdefault("VAEP_DIFF", "1")

from squadhan import train_e2e_vaep as T  # noqa: E402
from squadhan.config import (CHECKPOINTS_DIR, DROPOUT, HIDDEN_CHANNELS, METRICS_DIR,  # noqa: E402
                             NODE_DIM, NUM_HEADS, NUM_LAYERS)
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP  # noqa: E402
from experiments.min_minutes_eval import build_diff_ymap, fold_split, load_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-tag", required=True)
    ap.add_argument("--coord-skip", action="store_true")
    ap.add_argument("--seg-token", action="store_true")
    ap.add_argument("--no-gnn", action="store_true")
    ap.add_argument("--no-transformer", action="store_true")
    ap.add_argument("--value-skip", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    ymap = build_diff_ymap()
    gkids = T._build_gkids()
    rows = []
    for k in range(5):
        test_t, mu, sd = fold_split(T.SEASONS[k], T.SEED + k, ymap)
        test_s = load_test(test_t, gkids, ymap)
        m = E2ELineupOptimizerVAEP(
            node_dim=NODE_DIM, edge_dim=1, hidden=HIDDEN_CHANNELS,
            n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
            gk_select=True, no_gnn=args.no_gnn, no_transformer=args.no_transformer,
            seg_token=args.seg_token, coord_skip=args.coord_skip,
            value_skip=args.value_skip).to(device)
        ckpt = CHECKPOINTS_DIR / f"e2e_vaep_scalar{args.ckpt_tag}_stage1_fold{k}.pt"
        m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        m.eval()

        yt, yp, pos_t, pos_p = [], [], [], []
        with torch.no_grad():
            for d in test_s:
                gid, ih = int(d.game_id), int(d.is_home_game.view(-1).item())
                if (gid, ih) not in ymap:
                    continue
                pred_c, coords_c, _ = m(d.to(device), teacher_forcing=True)
                yp.append(float(pred_c.item()) * sd + mu)
                yt.append(ymap[(gid, ih)])
                gt = d.our_positions.view(-1, 2)
                pos_p.extend(coords_c.reshape(-1).cpu().numpy().tolist())
                pos_t.extend(gt.reshape(-1).cpu().numpy().tolist())

        rv = T._reg_metrics(yt, yp)
        rp = T._reg_metrics(pos_t, pos_p)
        rows.append({"fold": k, "test_season": T.SEASONS[k],
                     "s1_r2": rv["r2"], "s1_pearson": rv["pearson"],
                     "s1_rmse": rv["rmse"], "s1_nrmse": rv["nrmse"],
                     "s1_pos_r2": rp["r2"], "s1_pos_pearson": rp["pearson"],
                     "s1_pos_rmse": rp["rmse"], "s1_pos_nrmse": rp["nrmse"]})
        print(f"[{args.ckpt_tag} fold{k}] vaep_r2={rv['r2']:.4f} pos_r2={rp['r2']:.4f}", flush=True)

    out = METRICS_DIR / f"e2e_vaep_scalar{args.ckpt_tag}_test_cv.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print("→", out)


if __name__ == "__main__":
    main()
