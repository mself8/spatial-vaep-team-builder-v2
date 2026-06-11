#!/bin/bash
# ============================================================================
# One-shot reproduction of every number in the paper (Tables 1-3 + Fig. 3 case)
# ----------------------------------------------------------------------------
# Prerequisites: vaep/output/* (SPADL + VAEP artifacts) and
#                outputs/squad_graphs/*.pt must exist. If missing, build them:
#                  (1) python -m vaep.run_vaep
#                  (2) python -m squadhan.build_squad_dataset
#                (The VAEP label-model OOF AUC 0.91/0.92 from Sec. 4.3 is
#                printed in the training log of step (1).)
# Every run below performs 5-fold (leave-one-season-out) CV with
# Stage 1 + Stage 2 + test automatically, and is resumable per epoch
# (re-running the same command continues where it stopped).
# Results: outputs/metrics/e2e_vaep_scalar{RUN_TAG}_test_cv.csv
#          -- Table 2 = s1_* columns,
#             Table 3 'Ours'/'Coach' = s2_model_vaep / s2_coach_vaep columns
# ============================================================================
set -e
cd "$(dirname "$0")"
PY=${PY:-python}

# ── Common config used throughout the paper (final model = cskip+lc10+vskip) ─
#  EDGE_SCALAR=1        : collapse 12-zone edge vectors to a scalar
#  VAEP_DIFF=1          : target = our-team minus opponent VAEP (advantage)
#  GK_SELECT=1          : separate GK pool, top-1 selection
#  COORD_SKIP=1         : skip connection from encoder embedding to position head
#  VALUE_SKIP=1         : skip connection from mean encoder embedding to value head
#  LAMBDA_COORD=10      : auxiliary position-loss weight (lambda in Eq. 7)
#  MIN_ELIG_MINUTES=900 : eligibility filter -- exclude players with fewer than
#                         900 minutes over 2021-25 (duplicate player IDs merged)
COMMON="EDGE_SCALAR=1 VAEP_DIFF=1 GK_SELECT=1 COORD_SKIP=1 VALUE_SKIP=1 LAMBDA_COORD=10 MIN_ELIG_MINUTES=900"
TAG=_gksel_sc_lc10_diff_cskip_vskip

# ── Table 2 + Table 3 "SquadHAN (Ours)" / "Coach actual" ─────────────────────
env $COMMON RUN_TAG=$TAG \
  $PY -m squadhan.train_e2e_vaep --fold -1 --stage 0

# ── Table 2 ablation: no Transformer (same config + NO_TRANSFORMER=1) ────────
#    Table 2 measures Stage-1 quality, so train stage 1 only and score the
#    checkpoints with eval_stage1_cv.
env $COMMON NO_TRANSFORMER=1 RUN_TAG=${TAG}_notrf \
  $PY -m squadhan.train_e2e_vaep --fold -1 --stage 1
$PY -m experiments.eval_stage1_cv --ckpt-tag ${TAG}_notrf \
  --no-transformer --coord-skip --value-skip

# ── Table 2 ablation: no GNN (same config + NO_GNN=1) ────────────────────────
env $COMMON NO_GNN=1 RUN_TAG=${TAG}_nognn \
  $PY -m squadhan.train_e2e_vaep --fold -1 --stage 1
$PY -m experiments.eval_stage1_cv --ckpt-tag ${TAG}_nognn \
  --no-gnn --coord-skip --value-skip

# ── Table 2 baseline: XGBoost (independent baseline, same LOSO splits) ───────
$PY -m experiments.xgb_stage1_ablation

# ── Table 3 'Ours' re-scored with the common evaluator (consistency check) ───
env WINNER_TAG=$TAG WINNER_VALUE_SKIP=1 MIN_ELIG_MINUTES=900 \
  $PY -m experiments.ablation_common_eval

# ── Table 3 'Coach actual' cross-check (must equal the s2_coach_vaep column) ─
env COACH_EVAL_TAG=$TAG COACH_VALUE_SKIP=1 \
  $PY -m experiments.coach_eval

# ── Table 3 baseline: Team-Builder (same eligibility filter, scored by the ──
#    frozen Ours evaluator). TB_FORMATION=modal: fixes the most frequent broad
#    formation of the train split (4-3-3 in every fold).
env MIN_ELIG_MINUTES=900 TB_EVAL_TAG=$TAG TB_COORD_SKIP=1 TB_VALUE_SKIP=1 TB_FORMATION=modal \
  $PY -m experiments.teambuilder_baseline

# ── Table 1: metric vs match-points correlations ─────────────────────────────
#    (run compute_xg / compute_xt first if the xG/xT CSVs are missing)
$PY -m experiments.vaep_result_corr

# ── Sec. 4.3: provider-xG label-model AUC 0.80 ───────────────────────────────
$PY -m experiments.xg_shot_auc

# ── Sec. 4.6 / Figure 3: case-study search (paper case = fold 0) ─────────────
env CASE_TAG=$TAG CASE_VALUE_SKIP=1 \
  $PY -m experiments.case_search

echo "Done -- see *_test_cv.csv under outputs/metrics/."
