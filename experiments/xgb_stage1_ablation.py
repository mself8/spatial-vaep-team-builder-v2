"""
XGBoost Stage-1 evaluator ablation (Table 2).
Reuses the same 5-fold LOSO splits as train_e2e_vaep.py.

Input features: coach actual XI (GK node0 + our_starter_of_pool_idx × 10 OF nodes)
  + opponent 11 nodes. Each node is 48D VAEP-zone features.
  Pooling: mean + sum over our 11, mean over opp 11, plus is_home flag → 145D.

Target: VAEP advantage (our_total_VAEP - opp_total_VAEP), same as VAEP_DIFF mode.
Output: outputs/metrics/e2e_vaep_scalar_xgb_diff_test_cv.csv
"""
import random

import numpy as np
import pandas as pd
import torch
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import os
os.environ.setdefault("EDGE_SCALAR", "1")   # match train_e2e_vaep default for xgb

from squadhan.train_e2e_vaep import (
    _build_yvaep_map,
    _build_gkids,
    _reg_metrics,
    _load_samples_fold,
    SEASONS,
    SEED,
    METRICS_DIR,
)

OUT = METRICS_DIR / "e2e_vaep_scalar_xgb_diff_test_cv.csv"
HEADER = [
    "fold", "test_season",
    "s1_r2", "s1_pearson", "s1_rmse", "s1_nrmse",
    "s1_coord_mse",
    "s1_pos_r2", "s1_pos_pearson", "s1_pos_rmse", "s1_pos_nrmse",
    "s2_model_vaep", "s2_coach_vaep", "s2_delta_vaep", "s2_selection_acc",
]


def _extract_pos_features(data_list) -> tuple[np.ndarray, np.ndarray]:
    """Per-player: 48D individual features → (x, y) coord. Returns (N*11 × 48, N*11 × 2)."""
    X_rows, y_rows = [], []
    for d in data_list:
        our_x = d["our_squad"].x
        pid_all = d["our_squad"].player_ids
        idx_of = d.our_starter_of_pool_idx
        our_xi = torch.cat([our_x[0:1], our_x[idx_of + 1]], dim=0)  # (11, 48) node order
        # our_positions is sorted by ascending player_id (build_squad_dataset) → sort features the same way
        xi_pids = torch.cat([pid_all[0:1], pid_all[1:][idx_of]])     # (11,) pids in node order
        order = torch.argsort(xi_pids)
        our_xi = our_xi[order]                                       # pid-sorted → matches pos_gt
        pos_gt = d.our_positions.view(11, 2)  # (11, 2) normalized [0,1], pid-sorted
        for i in range(11):
            X_rows.append(our_xi[i].numpy())
            y_rows.append(pos_gt[i].numpy())
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


def _extract_features(data_list) -> tuple[np.ndarray, np.ndarray]:
    """Data list → (X: N×145, y: N,) arrays.

    our 11 = GK (node 0) + OF (our_starter_of_pool_idx, offset by +1 since node0=GK).
    """
    X_rows, y_rows = [], []
    for d in data_list:
        our_x = d["our_squad"].x          # (pool, 48)
        opp_x = d["opp"].x                # (11, 48)
        idx_of = d.our_starter_of_pool_idx  # LongTensor, 10 OF indices into node[1:]

        # GK = node 0, OF = idx_of + 1 (shift because node0 is GK)
        of_node_idx = idx_of + 1
        gk_feat = our_x[0:1]              # (1, 48)
        of_feat = our_x[of_node_idx]      # (10, 48)
        our_xi = torch.cat([gk_feat, of_feat], dim=0)  # (11, 48)

        our_mean = our_xi.mean(0).numpy()   # (48,)
        our_sum = our_xi.sum(0).numpy()     # (48,)
        opp_mean = opp_x.mean(0).numpy()    # (48,)
        is_home = np.array([float(getattr(d, "is_home", 0))])  # (1,)

        X_rows.append(np.concatenate([our_mean, our_sum, opp_mean, is_home]))
        y_rows.append(float(d._yv))

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float64)


def main():
    done_folds: set[int] = set()
    if OUT.exists():
        done_folds = set(pd.read_csv(OUT)["fold"].astype(int).tolist())
        print(f"[resume] already done folds: {sorted(done_folds)}")

    print("Building VAEP diff map…")
    ymap_raw = _build_yvaep_map()
    ymap: dict = {}
    for (gid, ih), val in ymap_raw.items():
        opp = ymap_raw.get((gid, 1 - ih))
        if opp is not None:
            ymap[(gid, ih)] = val - opp
    print(f"  VAEP diff entries: {len(ymap)}")

    gkids = _build_gkids()

    rows = []
    for k in range(5):
        if k in done_folds:
            print(f"[skip] fold {k}")
            continue

        ts = SEASONS[k]
        seed_k = SEED + k
        random.seed(seed_k); np.random.seed(seed_k)
        print(f"\n{'='*50}\nFold {k} — test={ts} (seed={seed_k})")

        train_s, _val_s, test_s = _load_samples_fold(ts, seed_k, ymap, gkids)
        print(f"  Train {len(train_s)}  Test {len(test_s)}")

        X_tr, y_tr = _extract_features(train_s)
        X_te, y_te = _extract_features(test_s)

        # Normalise target (same convention as GNN training)
        mu, sd = float(y_tr.mean()), float(y_tr.std() + 1e-8)
        y_tr_n = (y_tr - mu) / sd

        model = XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed_k, n_jobs=-1,
        )
        model.fit(X_tr, y_tr_n)

        # Predict on original scale
        y_pred = model.predict(X_te) * sd + mu

        m = _reg_metrics(y_te.tolist(), y_pred.tolist())
        print(f"  R²={m['r2']:.4f}  Pearson={m['pearson']:.4f}  "
              f"RMSE={m['rmse']:.4f}  NRMSE={m['nrmse']:.4f}")

        # Position prediction: per-player 48D → (x, y)
        X_pos_tr, y_pos_tr = _extract_pos_features(train_s)
        X_pos_te, y_pos_te = _extract_pos_features(test_s)
        pos_model = MultiOutputRegressor(XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=seed_k, n_jobs=-1,
        ))
        pos_model.fit(X_pos_tr, y_pos_tr)
        y_pos_pred = pos_model.predict(X_pos_te)  # (N_test*11, 2)
        coord_mse = float(np.mean((y_pos_pred - y_pos_te) ** 2))
        pm = _reg_metrics(y_pos_te.reshape(-1).tolist(), y_pos_pred.reshape(-1).tolist())
        print(f"  pos R²={pm['r2']:.4f}  RMSE={pm['rmse']:.4f}  NRMSE={pm['nrmse']:.4f}")

        rows.append({
            "fold": k, "test_season": ts,
            "s1_r2": m["r2"], "s1_pearson": m["pearson"],
            "s1_rmse": m["rmse"], "s1_nrmse": m["nrmse"],
            "s1_coord_mse": coord_mse,
            "s1_pos_r2": pm["r2"], "s1_pos_pearson": pm["pearson"],
            "s1_pos_rmse": pm["rmse"], "s1_pos_nrmse": pm["nrmse"],
            "s2_model_vaep": 0.0, "s2_coach_vaep": 0.0,
            "s2_delta_vaep": 0.0, "s2_selection_acc": 0.0,
        })

    if rows:
        df_new = pd.DataFrame(rows, columns=HEADER)
        if OUT.exists():
            df_old = pd.read_csv(OUT)
            df_new = pd.concat([df_old, df_new], ignore_index=True)
        METRICS_DIR.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(OUT, index=False)
        print(f"\nSaved → {OUT}")
        print(df_new[["fold", "test_season", "s1_r2", "s1_pearson", "s1_rmse", "s1_nrmse"]])


if __name__ == "__main__":
    main()
