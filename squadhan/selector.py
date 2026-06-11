"""Gumbel-Top-k selector — differentiable subset selection (trained in Stage 2).

A generic module that differentiably selects k players from a pool of arbitrary size N.
Used in e2e_model*.py in two roles:
  - of_selector : pick 10 from the OF pool (19 players) (k=10)
  - gk_selector : pick 1 from the GK pool (1-3 players) (k=1, when GK_SELECT=1)

⚠ Change in GK handling:
  Before (e2e_model.py): the GK was always fixed as node 0 → no gk_selector.
  Now (e2e_model_vaep.py): with GK_SELECT=1 the GK pool is split off via the is_gk
  mask and selected competitively by gk_selector (GumbelTopK) → fixes the bug where
  backup GKs were mixed into the OF pool.

Mechanism (Binary Concrete relaxation):
  1. score_v = MLP([emb_v ; opp_ctx]) — score conditioned on the opponent context
  2. During training: add logistic noise ε ~ Logistic(0,1) → ε̃ = (score + ε) / τ
     (τ = temperature, annealed 1.0→0.1)
  3. soft gate: σ_v = sigmoid(ε̃_v)  — the exact Binary Concrete relaxation
  4. hard gate: hard_v = 1 if v in TopK(ε̃) else 0
  5. STE mask:  m_v = hard_v − sg(σ_v) + σ_v
     → forward: m_v ∈ {0,1} (discrete selection)
     → backward: ∂m_v/∂score_v = ∂σ_v/∂score_v (smooth gradient)
"""

import torch
import torch.nn as nn


class GumbelTopK(nn.Module):
    """Differentiably select k players from the outfield pool.

    Parameters
    ----------
    in_dim : int
        Player embedding dimension (64)
    opp_dim : int
        Opponent context dimension (64)
    temperature : float
        Gumbel-Softmax temperature, annealed 1.0→0.1 over training.
    """

    def __init__(self, in_dim: int, opp_dim: int, temperature: float = 1.0):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim + opp_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, 1),
        )
        self.temperature = temperature

    def forward(self, x: torch.Tensor, opp_ctx: torch.Tensor, k: int,
                training: bool = True, elig: torch.Tensor = None):
        """
        Parameters
        ----------
        x       : (N, in_dim)   — our team's candidate player embeddings
        opp_ctx : (opp_dim,)    — opponent context (broadcast to each player)
        k       : int           — number of players to select
        training: bool          — if True add Gumbel noise; if False deterministic
        elig    : (N,) bool     — candidate eligibility mask (False = structurally excluded from top-k).
                                  None (default) keeps behavior 100% identical.

        Returns
        -------
        mask       : (N,)  — STE mask (forward=0/1, backward passes soft gradients)
        raw_logits : (N,)  — noise-free raw scores (for a selection loss; currently unused)
        idx        : (k,)  — indices of the k selected players (topk result)
        """
        # Concat the opponent context to each player embedding → score
        ctx = opp_ctx.unsqueeze(0).expand(x.size(0), -1)   # (N, opp_dim)
        raw_logits = self.score(torch.cat([x, ctx], dim=-1)).squeeze(-1)  # (N,)
        if elig is not None:
            # Push ineligible candidates to -1e9, excluding them from top-k (and the soft gate)
            raw_logits = raw_logits.masked_fill(~elig, -1e9)

        if training:
            # Logistic noise: Binary Concrete relaxation for k>1 subset selection
            u = torch.rand_like(raw_logits)
            noise = torch.log(u + 1e-10) - torch.log(1 - u + 1e-10)
            logits_n = (raw_logits + noise) / self.temperature
        else:
            logits_n = raw_logits

        # Binary Concrete: sigmoid (for k>1 subset selection)
        soft = torch.sigmoid(logits_n)

        # Hard top-k pick
        _, idx = logits_n.topk(k)
        hard = torch.zeros_like(logits_n)
        hard[idx] = 1.0

        # STE: hard in forward, soft gradient flow in backward
        mask = hard - soft.detach() + soft

        return mask, raw_logits, idx
