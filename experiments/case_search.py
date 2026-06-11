"""Case-study search (§4.6 / Figure 3) — the paper's case = gid 26658 (2021 Busan vs Seoul E-Land).

Hard filters: DF>=3/MF>=2/FW>=1, Δv̂>0.05, predicted vs actual sign agreement & |error|<2.5,
          legibility>=0.9 (depth matches nominal position), margin>=0.03 (vertical band separation).
Horizontal criteria: DF-line width spread>=0.45, XI mean width centered (|μ-0.5|<=0.10),
          both left (<=0.35) and right (>=0.65) occupied.
Output: outputs/metrics/case_candidates.csv + top rows as a console table.

Usage:  CASE_TAG=_gksel_sc_lc10_diff_cskip_vskip python -m experiments.case_search
"""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("EDGE_SCALAR", "1"); os.environ.setdefault("GK_SELECT", "1")
os.environ.setdefault("VAEP_DIFF", "1");  os.environ.setdefault("MIN_ELIG_MINUTES", "900")

import numpy as np, pandas as pd, torch
from squadhan import train_e2e_vaep as T
from squadhan.config import CHECKPOINTS_DIR, NODE_DIM, HIDDEN_CHANNELS, NUM_HEADS, NUM_LAYERS, DROPOUT, VAEP_OUTPUT_DIR
from squadhan.e2e_model_vaep import E2ELineupOptimizerVAEP

POS_BROAD = {"GK":"GK","CB":"DF","LB":"DF","RB":"DF","RWB":"DF","LWB":"DF",
             "CM":"MF","CAM":"MF","CDM":"MF","LM":"MF","RM":"MF",
             "CF":"FW","RW":"FW","LW":"FW","RF":"FW","LF":"FW"}
TAG = os.environ.get("CASE_TAG", "_gksel_sc_lc10_diff_cskip_vskip")
VALUE_SKIP = os.environ.get("CASE_VALUE_SKIP", "1") == "1"
OUT_CSV = ROOT / 'outputs' / 'metrics' / 'case_candidates.csv'

ymap = T._build_yvaep_map(); dmap = {}
for (g,ih),v in ymap.items():
    o = ymap.get((g,1-ih))
    if o is not None: dmap[(g,ih)] = v-o
gkids = T._build_gkids()

pl = pd.read_csv(VAEP_OUTPUT_DIR/'players.csv')
nick = pl.groupby('player_id')['nickname'].first().to_dict()
prim = (pl.dropna(subset=['starting_position_name'])
          .groupby('player_id')['starting_position_name']
          .agg(lambda s: s.value_counts().idxmax()).to_dict())
games = pd.read_csv(VAEP_OUTPUT_DIR/'games.csv')
gmeta = {int(r.game_id): r for r in games.itertuples(index=False)}
tname = pd.read_csv(VAEP_OUTPUT_DIR/'teams.csv').set_index('team_id')['team_name_ko'].to_dict()

rows = []
for k in range(5):
    season = T.SEASONS[k]
    tr, _, te = T._load_samples_fold(season, T.SEED + k, dmap, gkids)
    ys = np.array([d._yv for d in tr]); mu, sd = float(ys.mean()), float(ys.std()+1e-8)
    del tr
    m = E2ELineupOptimizerVAEP(node_dim=NODE_DIM, edge_dim=1, hidden=HIDDEN_CHANNELS,
        n_heads=NUM_HEADS, n_layers=NUM_LAYERS, dropout=DROPOUT,
        gk_select=True, no_gnn=False, coord_skip=True, value_skip=VALUE_SKIP)
    m.load_state_dict(torch.load(
        CHECKPOINTS_DIR/f'e2e_vaep_scalar{TAG}_stage2_fold{k}.pt',
        map_location='cpu', weights_only=False))
    m.eval()
    with torch.no_grad():
        for d in te:
            gid, ih = int(d.game_id), int(d.is_home_game.view(-1).item())
            actual = dmap.get((gid, ih))
            if actual is None: continue
            pid = d['our_squad'].player_ids
            gk_e = d['our_squad'].gk_elig; of_e = d['our_squad'].of_elig
            vc, _, _ = m(d, teacher_forcing=True)
            vm, cm, _ = m(d, teacher_forcing=False, gk_elig=gk_e, of_elig=of_e)
            vm_d = float(vm)*sd+mu; vc_d = float(vc)*sd+mu
            # agreement with the actual value (sign match + error)
            if np.sign(vm_d) != np.sign(actual) or abs(vm_d-actual) >= 2.5: continue
            delta = (float(vm)-float(vc))*sd
            if delta <= 0.05: continue
            our_emb, opp_emb = m.encoder(d); opp_ctx = opp_emb.mean(0)
            is_gk = d['our_squad'].is_gk.view(-1).bool()
            _,_,gi = m.gk_selector(our_emb[is_gk], opp_ctx, k=1, training=False, elig=gk_e)
            _,_,oi = m.of_selector(our_emb[~is_gk], opp_ctx, k=10, training=False, elig=of_e)
            gk_id = int(pid[is_gk][gi].item())
            xi = sorted([gk_id] + pid[~is_gk][oi].tolist())
            broads = {p: POS_BROAD.get(prim.get(int(p),''),'UNK') for p in xi}
            cnt = {b: sum(1 for p in xi if broads[p]==b and p!=gk_id) for b in ('DF','MF','FW','UNK')}
            if not (cnt['DF']>=3 and cnt['MF']>=2 and cnt['FW']>=1 and cnt['UNK']==0):
                continue
            depth = {p: float(cm[i,1]) for i,p in enumerate(xi)}
            width = {p: float(cm[i,0]) for i,p in enumerate(xi)}
            of_ids = [p for p in xi if p != gk_id]
            of_sorted = sorted(of_ids, key=lambda p: depth[p])
            want = ['DF']*cnt['DF'] + ['MF']*cnt['MF'] + ['FW']*cnt['FW']
            legi = sum(1 for p,wb in zip(of_sorted, want) if broads[p]==wb)/10.0
            if legi < 0.9: continue
            gm = {b: np.mean([depth[p] for p in of_ids if broads[p]==b]) for b in ('DF','MF','FW')}
            margin = min(gm['MF']-gm['DF'], gm['FW']-gm['MF'])
            if margin < 0.03: continue
            # ── horizontal criteria ──
            ws = np.array([width[p] for p in of_ids])
            df_w = np.array([width[p] for p in of_ids if broads[p]=='DF'])
            fw_w = np.array([width[p] for p in of_ids if broads[p]=='FW'])
            df_spread = float(df_w.max()-df_w.min())
            mean_off = abs(float(ws.mean())-0.5)
            if df_spread < 0.45 or mean_off > 0.10: continue
            if ws.min() > 0.35 or ws.max() < 0.65: continue
            hscore = df_spread + 0.5*(float(fw_w.max()-fw_w.min()) if len(fw_w)>1 else 0) - 2*mean_off
            coach_of = pid[1:][d.our_starter_of_pool_idx.view(-1).long()].tolist()
            coach_xi = sorted([int(pid[0])]+coach_of)
            n_swap = len([p for p in xi if p not in coach_xi])
            gmrow = gmeta[gid]
            our_tid = int(gmrow.home_team_id if ih else gmrow.away_team_id)
            opp_tid = int(gmrow.away_team_id if ih else gmrow.home_team_id)
            rows.append({
                'fold': k, 'season': season, 'gid': gid, 'side': 'home' if ih else 'away',
                'date': str(gmrow.game_date)[:10],
                'our': tname.get(our_tid,our_tid), 'opp': tname.get(opp_tid,opp_tid),
                'formation': f"{cnt['DF']}-{cnt['MF']}-{cnt['FW']}",
                'delta': round(delta,3), 'vc': round(vc_d,3), 'vm': round(vm_d,3),
                'actual': round(actual,3), 'err': round(abs(vm_d-actual),3),
                'legibility': legi, 'margin': round(margin,3),
                'df_spread': round(df_spread,3), 'mean_off': round(mean_off,3),
                'hscore': round(hscore,3), 'n_swap': n_swap,
                'gk_keep': int(gk_id)==int(pid[0]),
                'in': ','.join(nick.get(p,str(p)) for p in xi if p not in coach_xi),
                'out': ','.join(nick.get(p,str(p)) for p in coach_xi if p not in xi)})
    print(f"fold {k} ({season}) done — cumulative candidates {len(rows)}", flush=True)

df = pd.DataFrame(rows)
if len(df):
    df = df.sort_values(['hscore','margin','delta'], ascending=False)
df.to_csv(OUT_CSV, index=False)
print(f"\n{len(df)} candidates → {OUT_CSV}")
if len(df):
    cols = ['fold','season','gid','date','our','opp','formation','delta','vm','actual','err',
            'margin','df_spread','mean_off','hscore','n_swap','gk_keep']
    print(df[cols].head(15).to_string(index=False))
