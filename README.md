# Spatial Action Value Based Lineup Optimization

Code release for the CIKM 2026 submission *"Spatial Action Value Based Lineup Optimization"* (double-blind; this repository is anonymized).

The system recommends a starting XI from a match-day squad by combining
**spatial action values (VAEP)** with a **heterogeneous squad graph encoder
(SquadHAN)** and a **differentiable Top-k selector**, trained end-to-end in
two stages on five seasons of K League 1 & 2 event data.

---

## Key ideas

1. **VAEP + SquadHAN evaluator** — each player's action values (VAEP) are
   aggregated over a 12-zone pitch grid into node features; a heterogeneous
   graph encoder (SquadHAN: GATv2 per edge type + semantic attention) encodes
   intra-team cooperation (IO) and inter-team defensive response (ID).
2. **Differentiable subset selection** — 1 GK + 10 outfield players are
   selected without position labels via noisy Top-k relaxation with a
   Straight-Through Estimator, so selection gradients flow end-to-end.
3. **Two-stage training** — Stage 1 trains the evaluator (lineup → expected
   VAEP advantage regression, with an auxiliary position head) and freezes it.
   Stage 2 trains only the selector on top of the frozen evaluator to
   maximize predicted VAEP advantage.

## Pipeline at a glance

```
raw-data/ (Bepro event & lineup JSON, not included — see Data)
   │
   │  Step 1   python -m vaep.run_vaep
   ▼
vaep/output/          spadl_all.parquet · vaep_oof.parquet · games.csv · players.csv · teams.csv
   │
   │  Step 2   python -m squadhan.build_squad_dataset
   ▼
outputs/squad_graphs/ match_{gid}_{home|away}.pt   (per-match heterogeneous squad graphs)
   │
   │  Step 3   ./run_paper.sh   (5-fold LOSO: Stage 1 + Stage 2 + test, all paper configs)
   ▼
outputs/metrics/      *_test_cv.csv  (Table 2 = s1_* columns, Table 3 = s2_* columns)
outputs/checkpoints/  e2e_vaep_scalar{TAG}_stage{1|2}_fold{k}.pt
   │
   │  Step 4   jupyter notebook paper_results.ipynb
   ▼
Tables 1–3 + case study, exactly as reported in the paper
```

## Repository layout

```
.
├── README.md
├── requirements.txt          # tested package pins (torch 2.2.1 + PyG 2.7.0, CUDA 12.1)
├── run_paper.sh              # one-shot reproduction of every number in the paper
├── paper_results.ipynb       # executed notebook: Tables 1–3 + case study
├── player_id_groups.json     # groups of player_ids belonging to the same player (IDs only)
│
├── squadhan/                 # core model package
│   ├── config.py             # paths, hyperparameters, action-type constants
│   ├── zones.py              # 12-zone pitch partition (coordinates → zone)
│   ├── build_dataset.py      # VAEP → player graph feature helpers (IO/ID pair extraction)
│   ├── build_squad_dataset.py# per-match squad graph builder (static 2021–25 aggregation)
│   ├── squad_hgt.py          # SquadHAN encoder (GATv2 + semantic attention)
│   ├── selector.py           # differentiable Top-k subset selector (STE)
│   ├── e2e_model_vaep.py     # end-to-end lineup model (joint transformer + heads)
│   └── train_e2e_vaep.py     # two-stage training, 5-fold leave-one-season-out CV
│
├── vaep/                     # Stage 0: raw events → SPADL → VAEP
│   ├── run_vaep.py           # entry point (SPADL conversion + out-of-fold VAEP)
│   ├── core.py               # VAEP model training/prediction logic
│   └── lib/datatools/        # vendored data library (Bepro loader, SPADL schema, VAEP features)
│
└── experiments/              # scripts that reproduce the paper's tables and case study
    ├── vaep_result_corr.py   # Table 1 (metric ↔ match-points correlations)
    ├── compute_xg.py         # Table 1 prerequisite: provider xG per team-match
    ├── compute_xt.py         # Table 1 prerequisite: xT per team-match (socceraction)
    ├── xgb_stage1_ablation.py# Table 2: XGBoost evaluator baseline
    ├── eval_stage1_cv.py     # Table 2: scores stage1-only ablation checkpoints
    ├── min_minutes_eval.py   # shared eval helpers (eligibility, fold splits)
    ├── ablation_common_eval.py # Table 3 'Ours' re-scored with the common evaluator
    ├── coach_eval.py         # Table 3 'Coach actual' cross-check
    ├── teambuilder_baseline.py # Table 3 'Team-Builder' baseline (MILP, frozen evaluator)
    ├── xg_shot_auc.py        # §4.3 provider-xG label-model AUC
    ├── case_search.py        # §4.6 / Fig. 3 case-study search
    └── nb_helpers.py         # notebook helpers (model loading, recommendation, pitch plots)
```

`raw-data/`, `vaep/output/`, and `outputs/` are **not** included (data license; see below).

## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, `torch==2.2.1` (cu121) and `torch-geometric==2.7.0`
on a single NVIDIA RTX A6000. Install the torch/PyG wheels matching your CUDA
version first if the generic pip resolution fails.

## Data availability

The experiments use **private event and lineup data for K League 1 & 2
(2021–2025) licensed from Bepro**, which we are not permitted to
redistribute. No raw or derived match data is included in this repository.

What this means for reviewers:

- All code, configurations, and the exact commands for every reported number
  are included, and `paper_results.ipynb` is committed **with its executed
  outputs**, so every table in the paper can be inspected against the code
  that produced it.
- Anyone with access to the same Bepro K League feed can re-run the full
  pipeline by placing the JSON files in the layout below.

```
raw-data/
├── KLEAGUE1/                  # K League 1 (competition_id=587)
│   └── {2021..2025}/
│       ├── match.json         # season match list
│       ├── team.json
│       ├── player/            # player info JSON
│       └── match/{game_id}/
│           ├── lineup.json    # starters/bench + positions
│           ├── event_data.json
│           └── info.json
└── KLEAGUE2/                  # K League 2 (competition_id=588)
    └── {2021..2025}/          # same structure
```

`player_id_groups.json` (included) lists groups of `player_id`s that belong
to the same player across seasons/transfers — used only to merge minutes for
the 900-minute eligibility filter. It contains numeric IDs only.

## Reproducing the paper

### One-shot

```bash
python -m vaep.run_vaep                  # Step 1: SPADL + out-of-fold VAEP
python -m squadhan.build_squad_dataset   # Step 2: per-match squad graphs
./run_paper.sh                           # Step 3: every training/eval run in the paper
jupyter notebook paper_results.ipynb     # Step 4: render Tables 1–3 + case study
```

`run_paper.sh` runs the final configuration (`cskip+lc10+vskip`), both
ablations, the XGBoost baseline, the Team-Builder baseline, the coach
cross-check, Table 1 correlations, the xG label-model AUC, and the
case-study search. Every run is resumable (re-running the same command
continues from the last finished epoch/fold). Stage 1+2 training is roughly
a few hours per configuration on a single A6000.

### Main configuration (used everywhere in `run_paper.sh`)

```bash
COMMON="EDGE_SCALAR=1 VAEP_DIFF=1 GK_SELECT=1 COORD_SKIP=1 VALUE_SKIP=1 LAMBDA_COORD=10 MIN_ELIG_MINUTES=900"
env $COMMON RUN_TAG=_gksel_sc_lc10_diff_cskip_vskip \
  python -m squadhan.train_e2e_vaep --fold -1 --stage 0   # all 5 folds, Stage 1+2+test
```

| Env var | Default | Meaning |
|---|---|---|
| `RUN_TAG` | (none) | suffix for checkpoint/metric filenames |
| `GK_SELECT` | 0 | 1 = separate GK pool (top-1) and outfield pool (top-10) |
| `EDGE_SCALAR` | 0 | 1 = collapse 12-D zone edges to a 1-D scalar at load time |
| `LAMBDA_COORD` | 1 | weight of the auxiliary position loss (λ = 10 in Eq. 7) |
| `VAEP_DIFF` | 0 | 1 = target is our-team − opponent VAEP (advantage) |
| `COORD_SKIP` | 0 | 1 = skip connection from encoder embedding to position head |
| `VALUE_SKIP` | 0 | 1 = skip connection from mean encoder embedding to value head |
| `MIN_ELIG_MINUTES` | 0 | eligibility filter: exclude players below this many minutes (paper: 900) |
| `NO_GNN` | 0 | ablation: replace the GNN encoder with an MLP |
| `NO_TRANSFORMER` | 0 | ablation: remove the joint transformer |

Outputs: `outputs/metrics/e2e_vaep_scalar{TAG}_test_cv.csv` — **Table 2 =
`s1_*` columns; Table 3 'Ours'/'Coach' = `s2_model_vaep`/`s2_coach_vaep`**
(mean ± std over the 5 LOSO folds).

### Where each number in the paper comes from

| Paper | Numbers | Reproduced by |
|---|---|---|
| Table 1 | Pearson r +0.689 / +0.358 / +0.063 / +0.624 | `python -m experiments.vaep_result_corr` (after `compute_xg`, `compute_xt`) |
| Table 2 (Ours, ablations) | VAEP R² 0.165, position R² 0.673, … | `s1_*` columns of `e2e_vaep_scalar_gksel_sc_lc10_diff_cskip_vskip{,_notrf,_nognn}_test_cv.csv` (ablations scored by `experiments.eval_stage1_cv`) |
| Table 2 (XGBoost) | R² 0.123 / 0.616 | `python -m experiments.xgb_stage1_ablation` |
| Table 3 SquadHAN (Ours) | 0.829 ± 0.382, SelAcc 0.631 | `s2_model_vaep`, `s2_selection_acc` columns (cross-checked by `experiments.ablation_common_eval`) |
| Table 3 Coach actual | −0.020 ± 0.317 | `s2_coach_vaep` column (cross-checked by `experiments.coach_eval`) |
| Table 3 Team-Builder | 0.155 ± 0.340, SelAcc 0.630 | `TB_FORMATION=modal python -m experiments.teambuilder_baseline` (scored by the frozen Ours evaluator) |
| §4.3 label-model AUC | VAEP 0.91/0.92, xG 0.80 | `python -m vaep.run_vaep` training log; `python -m experiments.xg_shot_auc` |
| §4.6 / Fig. 3 case study | −0.26 → +0.42 (actual +0.62) | `python -m experiments.case_search` → row of the paper's match in `outputs/metrics/case_candidates.csv`; rendered in `paper_results.ipynb` |
| §4.1 data scale | 2,282 matches, 5 seasons | `vaep/output/games.csv`, `outputs/squad_graphs/*.pt` counts |

## Model

```
[our squad (≤20 candidates) + opponent starters (11)]
        nodes: 48-D VAEP-zone features · edges: 1-D scalar (collapsed from 12-D)
                          │
            SquadHAN encoder (GATv2 over 4 edge types + semantic attention, ×2)
                          │
              our_emb (20×64)        opp_emb (11×64)
                          │
        GK selector (top-1)  +  outfield selector (top-10)   ← noisy Top-k + STE
                          │
              selected XI (11×64)  ‖  opponent XI (11×64)
                          │
                joint transformer (22 tokens, 2 layers, 4 heads)
                          │
        ├── position head → predicted formation coordinates (11×2)  [Stage 1 aux]
        └── value head    → predicted VAEP advantage v̂ (scalar)

Stage 1:  loss = MSE(v̂, y) + λ·MSE(coords)        (evaluator, then frozen)
Stage 2:  loss = −v̂                                (selector only)
```

**Node features (48-D):** VAEP summed over 4 action groups × 12 pitch zones,
normalized per 90 minutes, aggregated over all five seasons (static skill
representation).

**Edge types:** **IO** (intra-team cooperation; consecutive same-team action
pairs, symmetric) and **ID** (inter-team defensive response; defender-side
VAEP accumulated on directed edges), each stored as 12-D zone vectors and
collapsed to a scalar when `EDGE_SCALAR=1`.
