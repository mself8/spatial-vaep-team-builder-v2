"""Squad graph encoder — SquadHAN.

Name in the paper: SquadHAN (Squad Heterogeneous Attention Network).
The code file is named squad_hgt.py, but the architecture follows the HAN paradigm.

HAN vs HGT:
  HGT (Hu et al. 2020): relation-specific W_K/W_Q/W_V + mutual attention.
  HAN (Wang et al. 2019): per-metapath node-level attention + semantic attention (weighted sum).
  SquadHAN follows HAN: GATv2 node-level attention (per edge type) +
  semantic attention (weighted sum across edge types). GATv2 replaces the GAT of the original HAN.

Position in the pipeline:
  build_squad_dataset.py → (HeteroData .pt) → e2e_model*.py calls SquadHGT (this module)
    → (our_emb: 20×64, opp_emb: 11×64) → selector → Transformer → heads

Input (HeteroData):
  our_squad.x     : (20, 48)  — our squad node features (GK1 + OF19)
  opp.x           : (11, 48)  — opponent starter node features
  4 edge types (12D edge features each; collapsed to 1D when EDGE_SCALAR=1):
    (our_squad, IO, our_squad) : cooperation pairs within our team (symmetric)
    (opp,       IO, opp)       : cooperation pairs within the opponent team (symmetric)
    (our_squad, ID, opp)       : our→opponent cross-team defensive reactions
    (opp,       ID, our_squad) : opponent→our cross-team defensive reactions

Output:
  our_emb : (20, hidden)  — our squad embeddings
  opp_emb : (11, hidden)  — opponent embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class _SemanticAttention(nn.Module):
    """Semantic attention that weight-sums the per-edge-type embeddings (HAN style).

    Per-type score: score_t = mean_i( q · tanh(W · z_i^t) )
    Weighted sum:   out = Σ_t softmax(score_t) · Z^t
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.W = nn.Linear(hidden, hidden)
        self.q = nn.Parameter(torch.zeros(hidden))

    def forward(self, z_list: list[torch.Tensor]) -> torch.Tensor:
        scores = []
        for z in z_list:
            e = torch.tanh(self.W(z))
            scores.append((e * self.q).sum(-1).mean())
        w = F.softmax(torch.stack(scores), dim=0)
        return sum(w_t * z_t for w_t, z_t in zip(w, z_list))


def _get_edge(data, rel: tuple):
    """Safely fetch edge_index and edge_attr. Returns (None, None) if there are no edges."""
    store = data[rel[0], rel[1], rel[2]]
    if not hasattr(store, "edge_index") or store.edge_index.size(1) == 0:
        return None, None
    return store.edge_index, getattr(store, "edge_attr", None)


class SquadHGT(nn.Module):
    """Our squad (N players) + opponent (11 players) → node embeddings.

    Reuses only the message-passing part of the original HGTPredictor; the classification head lives in the E2E model.
    """

    def __init__(self, node_dim: int, edge_dim: int, hidden: int,
                 num_layers: int, num_heads: int, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        head_dim = hidden // num_heads

        self.proj_our = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden))
        self.proj_opp = nn.Sequential(nn.Linear(node_dim, hidden), nn.LayerNorm(hidden))

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(nn.ModuleDict({
                "IO_OO": GATv2Conv(hidden, head_dim, heads=num_heads,
                                   edge_dim=edge_dim, dropout=dropout,
                                   add_self_loops=False),
                "IO_AA": GATv2Conv(hidden, head_dim, heads=num_heads,
                                   edge_dim=edge_dim, dropout=dropout,
                                   add_self_loops=False),
                "ID_OA": GATv2Conv(hidden, head_dim, heads=num_heads,
                                   edge_dim=edge_dim, dropout=dropout,
                                   add_self_loops=False),
                "ID_AO": GATv2Conv(hidden, head_dim, heads=num_heads,
                                   edge_dim=edge_dim, dropout=dropout,
                                   add_self_loops=False),
            }))

        self.sem_our = nn.ModuleList([_SemanticAttention(hidden) for _ in range(num_layers)])
        self.sem_opp = nn.ModuleList([_SemanticAttention(hidden) for _ in range(num_layers)])
        self.norms_our = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])
        self.norms_opp = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def forward(self, data):
        """
        Returns
        -------
        our_emb : (N, hidden)
        opp_emb : (11, hidden)
        """
        h = F.relu(self.proj_our(data["our_squad"].x))
        a = F.relu(self.proj_opp(data["opp"].x))

        for i, conv in enumerate(self.convs):
            ei_IO_OO, ea_IO_OO = _get_edge(data, ("our_squad", "IO", "our_squad"))
            ei_IO_AA, ea_IO_AA = _get_edge(data, ("opp", "IO", "opp"))
            ei_ID_OA, ea_ID_OA = _get_edge(data, ("our_squad", "ID", "opp"))
            ei_ID_AO, ea_ID_AO = _get_edge(data, ("opp", "ID", "our_squad"))

            z_IO_OO = (conv["IO_OO"](h, ei_IO_OO, ea_IO_OO)
                       if ei_IO_OO is not None else torch.zeros_like(h))
            z_IO_AA = (conv["IO_AA"](a, ei_IO_AA, ea_IO_AA)
                       if ei_IO_AA is not None else torch.zeros_like(a))
            z_ID_OA = (conv["ID_OA"]((h, a), ei_ID_OA, ea_ID_OA)
                       if ei_ID_OA is not None else torch.zeros_like(a))
            z_ID_AO = (conv["ID_AO"]((a, h), ei_ID_AO, ea_ID_AO)
                       if ei_ID_AO is not None else torch.zeros_like(h))

            h_new = self.sem_our[i]([z_IO_OO, z_ID_AO])
            a_new = self.sem_opp[i]([z_IO_AA, z_ID_OA])

            h = self.norms_our[i](h + F.dropout(h_new, p=self.dropout, training=self.training))
            a = self.norms_opp[i](a + F.dropout(a_new, p=self.dropout, training=self.training))

        return h, a
