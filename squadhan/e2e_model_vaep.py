"""End-to-end lineup recommendation model — VAEP-objective version (new file; the original e2e_model.py is untouched).

Differences from the original E2ELineupOptimizer:
  - result_head (3-class classification) → vaep_head (scalar regression): predicts the lineup's expected team VAEP.
  - forward output: (pred_vaep, coords(11,2), of_logits(19,))
  - Everything else (SquadHGT encoder, GumbelTopK selector, 22-player joint Transformer, coord_head) is reused unchanged.

Assumption: batch_size=1. our_squad index 0=GK, 1~19=OF (build_squad_dataset convention).
"""

import torch
import torch.nn as nn

from squadhan.squad_hgt import SquadHGT
from squadhan.selector import GumbelTopK


class _MLPEncoder(nn.Module):
    """no-GNN ablation encoder: per-node Linear+LayerNorm projection only (edges/attention ignored).

    Keeps the same inputs/outputs as SquadHGT (our_emb(20,h), opp_emb(11,h)) while
    removing GATv2 message passing and semantic attention → a pure "remove only the
    GNN" ablation.
    """

    def __init__(self, node_dim: int, hidden: int):
        super().__init__()
        self.proj_our = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())
        self.proj_opp = nn.Sequential(
            nn.Linear(node_dim, hidden), nn.LayerNorm(hidden), nn.ReLU())

    def forward(self, data):
        return self.proj_our(data["our_squad"].x), self.proj_opp(data["opp"].x)


class E2ELineupOptimizerVAEP(nn.Module):
    def __init__(self, node_dim: int = 48, edge_dim: int = 12,
                 hidden: int = 64, n_heads: int = 4, n_layers: int = 2,
                 dropout: float = 0.3, gk_select: bool = False,
                 no_gnn: bool = False, objective: str = "vaep",
                 seg_token: bool = False, no_transformer: bool = False,
                 coord_skip: bool = False, value_skip: bool = False,
                 n_trf_layers: int = None):
        super().__init__()
        self.hidden = hidden
        self.gk_select = gk_select
        self.no_gnn = no_gnn
        self.objective = objective
        self.seg_token = seg_token
        # no-Transformer ablation: bypass the joint Transformer in forward (selected
        # embeddings go straight to the heads). The module is still created — keeps
        # the stage2 freeze loop and state_dict compatible.
        self.no_transformer = no_transformer
        # COORD_SKIP=1: skip-concat raw encoder embeddings into the coordinate head input —
        # feeds individual position signals directly, undiluted by graph/Transformer
        # mixing (bypasses oversmoothing).
        self.coord_skip = coord_skip
        # VALUE_SKIP=1: skip-concat mean encoder embeddings (starting XI, opponent) into
        # the value head input — value-head version of coord_skip (feeds value signals
        # directly, undiluted by Transformer mixing).
        self.value_skip = value_skip
        # SEG_TOKEN=1: add our/opponent segment embeddings to the joint Transformer input
        # (previously there was no side token, so attention could tell teammates from
        # opponents only via the embedding distributions).
        # Default False — compatible with loading existing checkpoints.
        if seg_token:
            self.team_seg = nn.Embedding(2, hidden)

        if no_gnn:
            # no-GNN ablation: per-node Linear projection instead of SquadHGT (edges ignored).
            self.encoder = _MLPEncoder(node_dim=node_dim, hidden=hidden)
        else:
            self.encoder = SquadHGT(
                node_dim=node_dim, edge_dim=edge_dim, hidden=hidden,
                num_layers=n_layers, num_heads=n_heads, dropout=dropout,
            )

        self.of_selector = GumbelTopK(in_dim=hidden, opp_dim=hidden)
        # Competitive GK selection: top-1 from the GK pool. Not created when gk_select=False (checkpoint-compatible, resume-safe).
        if gk_select:
            self.gk_selector = GumbelTopK(in_dim=hidden, opp_dim=hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=n_heads,
            dim_feedforward=hidden * 4,
            dropout=dropout, batch_first=True,
        )
        # TRF_LAYERS: set the joint Transformer depth separately (default None = same n_layers as the encoder)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=(n_trf_layers or n_layers))

        self.coord_head = nn.Sequential(
            nn.Linear(hidden * 2 if coord_skip else hidden, 2),
            nn.Sigmoid(),
        )

        # VAEP head: [team_ctx(64), opp_ctx(64), is_home(1)] = 129D
        # with value_skip: + [starting-XI encoder mean(64), opponent encoder mean(64)] = 257D
        self.vaep_head = nn.Sequential(
            nn.Linear((hidden * 4 if value_skip else hidden * 2) + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # OBJECTIVE=points: same 129-D input → 3-class win/draw/loss logits.
        if objective == "points":
            self.result_head = nn.Sequential(
                nn.Linear(hidden + hidden + 1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            )

    def forward(self, data, teacher_forcing: bool = False,
                gk_elig: torch.Tensor = None, of_elig: torch.Tensor = None):
        """Single-graph forward (assumes batch_size=1).

        gk_elig / of_elig : bool eligibility masks in pool order (for inference; e.g. the minimum-minutes filter).
        None (default) keeps behavior 100% identical. Ignored under teacher_forcing.

        Returns
        -------
        out       : scalar predicted team VAEP (objective="vaep") or (3,) win/draw/loss logits (objective="points")
        coords    : (11, 2) — coordinates of the 11 starters (ascending player_id)
        of_logits : (19,)   — selector raw logits (currently unused)
        """
        # 1. HGT encoding
        our_emb, opp_emb = self.encoder(data)
        opp_ctx = opp_emb.mean(dim=0)

        player_ids = data["our_squad"].player_ids
        gk_emb = our_emb[0:1]                 # node 0 = starting GK (build convention)
        gk_id = player_ids[0:1]
        of_emb = our_emb[1:]
        of_ids = player_ids[1:]

        of_logits = None
        if teacher_forcing:
            # Teacher forcing: the coach's actual XI = node0 (GK) + our_starter_of_pool_idx (OF). Same regardless of gk_select.
            of_pool_idx = data.our_starter_of_pool_idx.view(-1).long()
            gk_sel_emb, gk_sel_ids = gk_emb, gk_id
            of_sel = of_emb[of_pool_idx]
            cur_of_ids = of_ids[of_pool_idx]
        elif self.gk_select:
            # Competitive GK selection: split GK/OF pools via the is_gk mask → GK top-1 + OF top-10 (structurally exactly 1 GK).
            is_gk = data["our_squad"].is_gk.view(-1).bool()
            gk_pool_emb, of_pool_emb = our_emb[is_gk], our_emb[~is_gk]
            gk_pool_ids, of_pool_ids = player_ids[is_gk], player_ids[~is_gk]
            gk_mask, _, gk_idx = self.gk_selector(gk_pool_emb, opp_ctx, k=1, training=self.training, elig=gk_elig)
            of_mask, of_logits, of_idx = self.of_selector(of_pool_emb, opp_ctx, k=10, training=self.training, elig=of_elig)
            gk_sel_emb = gk_pool_emb[gk_idx] * gk_mask[gk_idx].unsqueeze(1)
            gk_sel_ids = gk_pool_ids[gk_idx]
            of_sel = of_pool_emb[of_idx] * of_mask[of_idx].unsqueeze(1)
            cur_of_ids = of_pool_ids[of_idx]
        else:
            # Original: fixed GK (node0) + OF 19→10 Gumbel-Top-k.
            of_mask, of_logits, of_idx = self.of_selector(of_emb, opp_ctx, k=10, training=self.training, elig=of_elig)
            gk_sel_emb, gk_sel_ids = gk_emb, gk_id
            of_sel = of_emb[of_idx] * of_mask[of_idx].unsqueeze(1)
            cur_of_ids = of_ids[of_idx]

        # Sort the 11 starters by ascending player_id
        starter_ids = torch.cat([gk_sel_ids, cur_of_ids])
        sort_order = torch.argsort(starter_ids)
        starter_emb = torch.cat([gk_sel_emb, of_sel], dim=0)[sort_order]

        # 6. Joint Transformer (our 11 + opponent 11 = 22)
        joint_input = torch.cat([starter_emb, opp_emb], dim=0)
        if self.seg_token:
            seg = torch.cat([
                self.team_seg.weight[0].expand(starter_emb.size(0), -1),
                self.team_seg.weight[1].expand(opp_emb.size(0), -1)], dim=0)
            joint_input = joint_input + seg
        if self.no_transformer:
            joint_out = joint_input          # ablation: selected embeddings go straight to the heads
        else:
            joint_out = self.transformer(joint_input.unsqueeze(0)).squeeze(0)
        our_out = joint_out[:11]
        opp_out = joint_out[11:]

        # 7. Heads
        if self.coord_skip:
            coords = self.coord_head(torch.cat([our_out, starter_emb], dim=-1))
        else:
            coords = self.coord_head(our_out)
        team_ctx = our_out.mean(dim=0)
        opp_ctx_refined = opp_out.mean(dim=0)
        is_home = data.is_home_game.view(-1).to(team_ctx.dtype)[:1]
        if self.value_skip:
            vaep_input = torch.cat([team_ctx, opp_ctx_refined,
                                    starter_emb.mean(dim=0), opp_emb.mean(dim=0),
                                    is_home], dim=-1)
        else:
            vaep_input = torch.cat([team_ctx, opp_ctx_refined, is_home], dim=-1)
        if self.objective == "points":
            out = self.result_head(vaep_input)             # (3,) win/draw/loss logits
        else:
            out = self.vaep_head(vaep_input).squeeze()     # scalar () predicted team VAEP

        return out, coords, of_logits
