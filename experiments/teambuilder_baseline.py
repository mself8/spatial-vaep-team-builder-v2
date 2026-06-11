"""Team-Builder baseline (Cao et al., IEEE TVCG 2023) — K-League reimplementation (Table 3 in the paper).

Team-Builder is prior work that recommends lineups via ILP. Objective (Eq.1):
    V = λ1·VI + λ2·IO + λ3·ID    (scalar sums; our 12-zone × 4-action 48D representation is unused)
  VI_p  = individual VAEP   (sum over the 48D node features, our_squad.x.sum)
  IO_pq = same-team offensive interaction (scalar sum of IO edges, undirected)
  ID_p  = sum of defensive interactions against the 11 opponents (ID edges)
Selection: maximize GK 1 + OF 10 via ILP (scipy.milp) under the coach's actual broad formation (DF/MF/FW counts) constraint.
  - Scale correction: standardize VI/IO/ID by per-match std before λ weighting.
  - λ: as in the original paper, "estimated from historical match data" — per fold, regress match
    VAEP advantage on the actual XI's (VI,IO,ID) sums (z-scored) over train matches → clip negative coefficients → normalize to sum=1.
    (TB_LAM_FIXED=1 falls back to the fixed λ=1/3.)
  - If the formation cannot be filled, relax the constraint (drop the count constraints).
Evaluation: score the recommended XI with that fold's frozen Ours (SquadHAN) Stage-1 evaluator (VAEP_DIFF sd correction) →
      Model VAEP adv + SelAcc (recall of the coach's OF). 5-fold LOSO.

Run: python -m experiments.teambuilder_baseline  (from the repository root)
Output: outputs/metrics/e2e_vaep_scalar_teambuilder_diff_test_cv.csv  (fold, season, model_vaep, selacc)
Inputs (checkpoints, graphs, VAEP) are not modified.
"""
import os

os.environ.setdefault("EDGE_SCALAR", "1")
os.environ.setdefault("OBJECTIVE", "vaep")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import milp, LinearConstraint, Bounds

from squadhan.train_e2e_vaep import (
    _build_yvaep_map, _build_gkids, _load_samples_fold, SEASONS, SEED,
    CHECKPOINTS_DIR, METRICS_DIR,
)
from experiments.nb_helpers import load_model, player_meta, POS_BROAD

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAM = (1 / 3, 1 / 3, 1 / 3)                 # (λ1=VI, λ2=IO, λ3=ID) fixed fallback (TB_LAM_FIXED=1)
LAM_FIXED = os.environ.get("TB_LAM_FIXED", "0") == "1"
# If MIN_ELIG_MINUTES>0, the loader attaches eligibility masks (train_e2e_vaep._attach_elig)
# and the ILP excludes ineligible candidates → saved to a separate output file (keeps the unfiltered results)
_ME = float(os.environ.get("MIN_ELIG_MINUTES", "0"))
# TB_EVAL_TAG: run tag of the frozen evaluator used for scoring (to re-score when the main model changes)
TB_EVAL_TAG = os.environ.get("TB_EVAL_TAG", "_gksel_sc_lc1_diff")
TB_COORD_SKIP = os.environ.get("TB_COORD_SKIP", "0") == "1"
TB_VALUE_SKIP = os.environ.get("TB_VALUE_SKIP", "0") == "1"   # for loading the vskip evaluator
# TB_FORMATION: source of the ILP formation constraint — actual (coach's actual per match, default) | modal
# (most frequent broad formation in the fold's train split; no coach info at test time) | fixed "D-M-F" counts, e.g. 433
TB_FORMATION = os.environ.get("TB_FORMATION", "actual")
_FORM_SUF = "" if TB_FORMATION == "actual" else f"_form{TB_FORMATION}"
if TB_EVAL_TAG == "_gksel_sc_lc1_diff":
    OUT = METRICS_DIR / (
        f"e2e_vaep_scalar_teambuilder_diff_elig{int(_ME)}{_FORM_SUF}_test_cv.csv" if _ME > 0
        else f"e2e_vaep_scalar_teambuilder_diff{_FORM_SUF}_test_cv.csv")
else:
    OUT = METRICS_DIR / (
        f"e2e_vaep_scalar_teambuilder{TB_EVAL_TAG}"
        f"{'_elig' + str(int(_ME)) if _ME > 0 else ''}{_FORM_SUF}_test_cv.csv")

_, _primary = player_meta()


def _broad(pid):
    return POS_BROAD.get(_primary.get(int(pid), ""), "MF")   # unknown positions default to MF


@torch.no_grad()
def score_xi(evaluator, data, gk_id, of_ids, mu, sd):
    """Score the recommended GK+OF 11 with the Ours evaluator (replicates the forward teacher-forcing path)."""
    our_emb, opp_emb = evaluator.encoder(data)
    pid = data["our_squad"].player_ids.cpu().numpy().tolist()
    node = {p: i for i, p in enumerate(pid)}
    gk_emb = our_emb[node[int(gk_id)]:node[int(gk_id)] + 1]
    of_emb = our_emb[[node[int(i)] for i in of_ids]]
    sids = torch.tensor([int(gk_id)] + [int(i) for i in of_ids], device=our_emb.device)
    order = torch.argsort(sids)
    starter = torch.cat([gk_emb, of_emb], 0)[order]
    jo = evaluator.transformer(torch.cat([starter, opp_emb], 0).unsqueeze(0)).squeeze(0)
    is_home = data.is_home_game.view(-1).to(jo.dtype)[:1]
    if getattr(evaluator, "value_skip", False):   # vskip evaluator: replicate the encoder-mean skip input
        vin = torch.cat([jo[:11].mean(0), jo[11:].mean(0),
                         starter.mean(0), opp_emb.mean(0), is_home], -1)
    else:
        vin = torch.cat([jo[:11].mean(0), jo[11:].mean(0), is_home], -1)
    v = evaluator.vaep_head(vin).squeeze()
    return float(v.item()) * sd + mu


def lineup_components(data):
    """(VI, IO, ID) sums of the actual (coach) XI — features for the λ regression."""
    x = data["our_squad"].x.cpu().numpy()
    xi = np.concatenate([[0], 1 + data.our_starter_of_pool_idx.view(-1).cpu().numpy()])
    xi_set = set(int(i) for i in xi)
    vi = float(x[xi].sum())
    io_store = data[("our_squad", "IO", "our_squad")]
    ei, ea = io_store.edge_index.cpu().numpy(), io_store.edge_attr.view(-1).cpu().numpy()
    io = float(sum(ea[k] for k in range(ei.shape[1])
                   if ei[0, k] in xi_set and ei[1, k] in xi_set)) / 2.0   # stored in both directions → /2
    id_store = data[("our_squad", "ID", "opp")]
    eid, ead = id_store.edge_index.cpu().numpy(), id_store.edge_attr.view(-1).cpu().numpy()
    idv = float(sum(ead[k] for k in range(eid.shape[1]) if eid[0, k] in xi_set))
    return vi, io, idv


def fit_lambda(train_s):
    """λ estimation as in the original paper: train actual-XI components (z-scored) → linear regression on VAEP advantage."""
    F = np.array([lineup_components(d) for d in train_s], dtype=np.float64)
    y = np.array([d._yv for d in train_s], dtype=np.float64)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)
    X = np.column_stack([Fz, np.ones(len(y))])
    beta = np.linalg.lstsq(X, y, rcond=None)[0][:3]
    lam = np.clip(beta, 0.0, None)
    lam = lam / lam.sum() if lam.sum() > 1e-12 else np.array([1/3, 1/3, 1/3])
    return tuple(float(v) for v in lam)


def coach_broad_counts(data):
    """Broad formation counts {DF/MF/FW: n} of the coach's actual XI for that match."""
    pid = data["our_squad"].player_ids.cpu().numpy()
    counts = {}
    for n in (1 + data.our_starter_of_pool_idx.view(-1).cpu().numpy()):
        c = _broad(pid[n])
        if c == "GK":
            c = "MF"
        counts[c] = counts.get(c, 0) + 1
    return counts


def resolve_formation(train_s):
    """Resolve TB_FORMATION → fixed-count dict, or None (= per-match actual)."""
    if TB_FORMATION == "actual":
        return None
    if TB_FORMATION == "modal":            # most frequent broad formation in the fold's train split (no test info)
        from collections import Counter
        modes = Counter(tuple(sorted(coach_broad_counts(d).items())) for d in train_s)
        return dict(modes.most_common(1)[0][0])
    d, m, f = (int(ch) for ch in TB_FORMATION)   # e.g. "433" → DF4 MF3 FW3
    assert d + m + f == 10, f"TB_FORMATION={TB_FORMATION}: counts must sum to 10"
    return {"DF": d, "MF": m, "FW": f}


def teambuilder_pick(data, lam=LAM, fixed_counts=None):
    """Select GK 1 + OF 10 via ILP → (gk_id, [of_ids])."""
    x = data["our_squad"].x.cpu().numpy()
    pid = data["our_squad"].player_ids.cpu().numpy()
    is_gk = data["our_squad"].is_gk.view(-1).cpu().numpy().astype(bool)
    N = len(pid)
    VI = x.sum(1)
    IO = np.zeros((N, N))
    io = data[("our_squad", "IO", "our_squad")]
    ei, ea = io.edge_index.cpu().numpy(), io.edge_attr.view(-1).cpu().numpy()
    for k in range(ei.shape[1]):
        a, b = ei[0, k], ei[1, k]
        IO[a, b] += ea[k]
        IO[b, a] += ea[k]
    ID = np.zeros(N)
    idd = data[("our_squad", "ID", "opp")]
    eid, ead = idd.edge_index.cpu().numpy(), idd.edge_attr.view(-1).cpu().numpy()
    for k in range(eid.shape[1]):
        ID[eid[0, k]] += ead[k]

    def z(a):
        s = a.std()
        return a / s if s > 1e-8 else a
    VIz, IDz = z(VI), z(ID)
    iostd = IO[IO != 0].std() if (IO != 0).any() else 1.0
    IOz = IO / iostd if iostd > 1e-8 else IO

    coach_counts = fixed_counts if fixed_counts is not None else coach_broad_counts(data)
    pos = np.array([_broad(p) for p in pid])

    pairs = [(i, j) for i in range(N) for j in range(i + 1, N) if IOz[i, j] != 0]
    nP = len(pairs)
    n_var = N + nP
    c = np.zeros(n_var)
    c[:N] = -(lam[0] * VIz + lam[2] * IDz)
    for t, (i, j) in enumerate(pairs):
        c[N + t] = -(lam[1] * IOz[i, j])

    A_gk = np.zeros(n_var); A_gk[:N] = is_gk.astype(float)
    A_tot = np.zeros(n_var); A_tot[:N] = 1.0
    base = [LinearConstraint(A_gk, 1, 1), LinearConstraint(A_tot, 11, 11)]
    ylin = []
    for t, (i, j) in enumerate(pairs):
        a1 = np.zeros(n_var); a1[N + t] = 1; a1[i] = -1; ylin.append(LinearConstraint(a1, -np.inf, 0))
        a2 = np.zeros(n_var); a2[N + t] = 1; a2[j] = -1; ylin.append(LinearConstraint(a2, -np.inf, 0))
        a3 = np.zeros(n_var); a3[N + t] = 1; a3[i] = -1; a3[j] = -1; ylin.append(LinearConstraint(a3, -1, np.inf))

    form = []
    feasible = True
    for cpos, cnt in coach_counts.items():
        sel = ((pos == cpos) & (~is_gk)).astype(float)
        if sel.sum() < cnt:
            feasible = False
            break
        A = np.zeros(n_var); A[:N] = sel
        form.append(LinearConstraint(A, cnt, cnt))

    # Minimum-minutes eligibility filter: if the loader attached masks, cap ineligible candidate variables at 0
    ub = np.ones(n_var)
    gk_e = getattr(data["our_squad"], "gk_elig", None)
    of_e = getattr(data["our_squad"], "of_elig", None)
    if gk_e is not None and of_e is not None:
        elig_nodes = np.ones(N, dtype=bool)
        elig_nodes[is_gk] = gk_e.cpu().numpy().astype(bool)
        elig_nodes[~is_gk] = of_e.cpu().numpy().astype(bool)
        ub[:N] = elig_nodes.astype(float)

    cons = base + ylin + (form if feasible else [])
    res = milp(c=c, constraints=cons, integrality=np.ones(n_var), bounds=Bounds(0, ub))
    if not res.success:                                   # retry with formation constraints relaxed
        res = milp(c=c, constraints=base + ylin, integrality=np.ones(n_var), bounds=Bounds(0, ub))

    xsel = np.where(res.x[:N] > 0.5)[0]
    gk_node = [n for n in xsel if is_gk[n]][0]
    of_nodes = [n for n in xsel if not is_gk[n]]
    return int(pid[gk_node]), [int(pid[n]) for n in of_nodes]


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
        ours = load_model(CHECKPOINTS_DIR / f"e2e_vaep_scalar{TB_EVAL_TAG}_stage2_fold{k}.pt",
                          gk_select=True, no_gnn=False, coord_skip=TB_COORD_SKIP,
                          value_skip=TB_VALUE_SKIP, device=device)
        lam_k = LAM if LAM_FIXED else fit_lambda(train_s)
        print(f"fold{k} λ̂ = (VI {lam_k[0]:.3f}, IO {lam_k[1]:.3f}, ID {lam_k[2]:.3f})", flush=True)
        fixed_counts = resolve_formation(train_s)
        if fixed_counts is not None:
            print(f"fold{k} formation[{TB_FORMATION}] = {fixed_counts}", flush=True)
        tb_vaep, sel = [], []
        for data in test_s:
            data = data.to(device)
            gk_id, of_ids = teambuilder_pick(data, lam_k, fixed_counts)
            tb_vaep.append(score_xi(ours, data, gk_id, of_ids, mu, sd))
            pid = data["our_squad"].player_ids
            coach_of = set(pid[1:][data.our_starter_of_pool_idx.view(-1).long()].cpu().numpy().tolist())
            sel.append(len(set(of_ids) & coach_of) / 10.0)
        r = {"fold": k, "test_season": ts,
             "s2_model_vaep": float(np.mean(tb_vaep)), "s2_selection_acc": float(np.mean(sel)),
             "lam_vi": lam_k[0], "lam_io": lam_k[1], "lam_id": lam_k[2]}
        rows.append(r)
        print(f"fold{k} {ts}: Team-Builder VAEP={r['s2_model_vaep']:+.3f}  SelAcc={r['s2_selection_acc']:.3f}",
              flush=True)

    df = pd.DataFrame(rows)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    v = df["s2_model_vaep"].values
    s = df["s2_selection_acc"].values
    print(f"\nSaved -> {OUT}")
    print(f"Model VAEP adv : {v.mean():+.3f} ± {v.std(ddof=1):.3f}")
    print(f"SelAcc         : {s.mean():.3f} ± {s.std(ddof=1):.3f}")


if __name__ == "__main__":
    main()
