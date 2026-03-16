"""
models.py — v5: LSTM and TFT with cross-asset attention for ranking.

Key improvements over v4:
  1. Cross-Asset Self-Attention — enables cross-sectional comparison
  2. Reduced capacity (hidden_dim=32, 1 LSTM layer) — fight overfitting
  3. Temporal position embeddings — encode time step position
  4. LightVSN — parameter-efficient Variable Selection Network
  5. Designed for ListNet ranking loss (outputs scores, not calibrated returns)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════════════
# Building blocks
# ═══════════════════════════════════════════════════════════════════════

class GatedLinearUnit(nn.Module):
    """GLU(x) = σ(W₁x+b₁) ⊙ (W₂x+b₂)."""

    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_out)
        self.fc2 = nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.sigmoid(self.fc1(x)) * self.fc2(x)


class GatedResidualNetwork(nn.Module):
    """GRN with skip connection, optional context, and LayerNorm."""

    def __init__(self, d_in, d_hidden, d_out=None, context_dim=None, dropout=0.3):
        super().__init__()
        d_out = d_out or d_in
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.context_fc = nn.Linear(context_dim, d_hidden, bias=False) if context_dim else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = GatedLinearUnit(d_out, d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None

    def forward(self, a, context=None):
        residual = self.skip(a) if self.skip else a
        x = self.fc1(a)
        if self.context_fc is not None and context is not None:
            x = x + self.context_fc(context)
        x = F.elu(x)
        x = self.dropout(self.fc2(x))
        x = self.glu(x)
        return self.layer_norm(residual + x)


class LightVariableSelectionNetwork(nn.Module):
    """
    Lightweight VSN: shared transform + pooled weight computation.

    Much fewer parameters than original TFT VSN (which flattens all
    M*d inputs into a single GRN). Here we pool over variables first,
    then compute weights from the pooled representation + optional context.

    v4 VSN params (M=33, d=32): ~55K
    v5 LightVSN params (M=33, d=32): ~5K
    """

    def __init__(self, n_vars, d_model, dropout=0.3, context_dim=None):
        super().__init__()
        self.var_transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        weight_in = d_model + (context_dim if context_dim else 0)
        self.weight_net = nn.Sequential(
            nn.Linear(weight_in, d_model),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_vars),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, inputs, context=None):
        """
        Parameters
        ----------
        inputs : (batch, n_vars, d_model)
        context : (batch, d_model) optional — static context (asset embedding)

        Returns
        -------
        selected : (batch, d_model) — weighted combination of transformed variables
        weights  : (batch, n_vars) — softmax feature importance weights
        """
        processed = self.var_transform(inputs)  # (B, M, d)

        pooled = inputs.mean(dim=1)  # (B, d)
        if context is not None:
            pooled = torch.cat([pooled, context], dim=-1)  # (B, d+c)

        weights = torch.softmax(self.weight_net(pooled), dim=-1)  # (B, M)
        selected = (weights.unsqueeze(-1) * processed).sum(dim=1)  # (B, d)
        return self.norm(selected), weights


class InterpretableMultiHeadAttention(nn.Module):
    """TFT-style attention: shared value projection, averaged heads."""

    def __init__(self, d_model, n_heads, dropout=0.3):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, self.d_k)   # Shared across heads
        self.W_o = nn.Linear(self.d_k, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        B, T, _ = q.size()
        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v)  # (B, T, d_k) — shared

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = self.dropout(torch.softmax(scores, dim=-1))

        V_exp = V.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        context = torch.matmul(attn, V_exp).mean(dim=1)  # average heads

        return self.W_o(context), attn.mean(dim=1)


class CrossAssetAttention(nn.Module):
    """
    Self-attention across assets at the same time step.

    This is the key v5 addition: after temporal encoding, each asset
    has a hidden representation. Cross-asset attention allows the model
    to compare assets directly, enabling cross-sectional ranking.

    Input:  h (N, d) — all N assets' representations at one time step
    Output: h (N, d) — representations enriched with cross-asset context
    """

    def __init__(self, d_model, n_heads=2, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        """h: (N, d) → (N, d) with cross-asset interaction."""
        h_3d = h.unsqueeze(0)  # (1, N, d) — batch=1, seq=N
        out, _ = self.attn(h_3d, h_3d, h_3d)
        return self.norm(h_3d + self.dropout(out)).squeeze(0)  # (N, d)


# ═══════════════════════════════════════════════════════════════════════
# LSTM with Cross-Asset Attention
# ═══════════════════════════════════════════════════════════════════════

class LSTMBaseline(nn.Module):
    """
    v5 LSTM: temporal encoding → cross-asset attention → score.

    Changes from v4:
      - Added temporal position embeddings
      - Added cross-asset self-attention after LSTM
      - Reduced to 1 LSTM layer (was 2)
      - Simplified head to single Linear (ranking doesn't need calibration)
      - Increased dropout (0.15 → 0.30)

    v4 params: ~82K → v5 params: ~18K
    """

    def __init__(self, n_features, n_assets, hidden_dim=32, n_layers=1,
                 dropout=0.30):
        super().__init__()
        self.input_proj = nn.Linear(n_features, hidden_dim)
        self.pos_emb = nn.Parameter(torch.randn(64, hidden_dim) * 0.02)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, n_layers,
                            batch_first=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.asset_emb = nn.Embedding(n_assets, hidden_dim)
        self.cross_attn = CrossAssetAttention(hidden_dim, 2, dropout)
        self.head = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, asset_idx):
        """
        x : (N, L, M) — all assets' feature sequences at one time step
        asset_idx : (N,) — asset identifiers
        """
        B, L, M = x.size()
        pos = self.pos_emb[:L]  # (L, d)
        h = self.dropout(self.input_proj(x)) + pos  # (N, L, d)
        _, (h_n, _) = self.lstm(h)
        h = h_n[-1] + self.asset_emb(asset_idx)  # (N, d)
        h = self.cross_attn(h)  # (N, d) — cross-asset comparison
        return self.head(h).squeeze(-1)  # (N,)

    def get_feature_importance(self):
        """Not available for LSTM."""
        return None


# ═══════════════════════════════════════════════════════════════════════
# TFT with Cross-Asset Attention
# ═══════════════════════════════════════════════════════════════════════

class TemporalFusionTransformer(nn.Module):
    """
    v5 Encoder-only TFT with cross-asset attention for ranking.

    Changes from v4:
      - LightVSN replaces heavy VSN (55K → 5K params for 33 features)
      - Cross-asset attention after temporal processing
      - Temporal position embeddings
      - Reduced dimensions (d=32, 1 LSTM layer, 2 heads)
      - Simplified output (no output GRN, just Linear)

    v4 params: ~209-368K → v5 params: ~28-42K
    """

    def __init__(self, n_features, n_assets, d_model=32, n_heads=2,
                 lstm_layers=1, dropout=0.30):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model

        # ── 1. Feature embeddings (vectorised) ──
        self.feat_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.feat_bias = nn.Parameter(torch.zeros(n_features, d_model))

        # ── 2. Temporal position embedding ──
        self.pos_emb = nn.Parameter(torch.randn(64, d_model) * 0.02)

        # ── 3. Asset embedding → static context for VSN ──
        self.asset_emb = nn.Embedding(n_assets, d_model)

        # ── 4. Light Variable Selection Network ──
        self.vsn = LightVariableSelectionNetwork(
            n_features, d_model, dropout=dropout, context_dim=d_model
        )

        # ── 5. LSTM encoder ──
        self.lstm = nn.LSTM(d_model, d_model, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0)
        self.post_lstm_gate = GatedLinearUnit(d_model, d_model)
        self.post_lstm_norm = nn.LayerNorm(d_model)

        # ── 6. Temporal multi-head attention ──
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.post_attn_gate = GatedLinearUnit(d_model, d_model)
        self.post_attn_norm = nn.LayerNorm(d_model)

        # ── 7. Cross-asset attention (NEW in v5) ──
        self.cross_attn = CrossAssetAttention(d_model, n_heads, dropout)

        # ── 8. Output ──
        self.output_fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

        # Store last VSN weights for interpretability
        self._last_vsn_weights = None

    def forward(self, x, asset_idx):
        """
        x : (N, L, M) — all assets' feature sequences at one time step
        asset_idx : (N,) — asset identifiers
        """
        B, L, M = x.size()

        # Static context from asset embedding
        static = self.asset_emb(asset_idx)  # (B, d)

        # Feature embedding: (B, L, M) → (B, L, M, d)
        embedded = x.unsqueeze(-1) * self.feat_weight + self.feat_bias

        # Position embedding
        pos = self.pos_emb[:L]  # (L, d)

        # VSN: flatten time into batch → (B*L, M, d)
        emb_flat = embedded.reshape(B * L, M, self.d_model)
        static_rep = static.unsqueeze(1).expand(-1, L, -1).reshape(B * L, self.d_model)
        selected, weights = self.vsn(emb_flat, static_rep)

        temporal = selected.reshape(B, L, self.d_model) + pos  # add position
        self._last_vsn_weights = weights.reshape(B, L, M)

        # LSTM encoder with gated skip
        lstm_out, _ = self.lstm(temporal)
        lstm_out = self.post_lstm_norm(
            temporal + self.post_lstm_gate(lstm_out)
        )

        # Temporal multi-head attention with gated skip
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = self.post_attn_norm(
            lstm_out + self.post_attn_gate(attn_out)
        )

        # Last time step → cross-asset attention
        final = attn_out[:, -1, :]  # (B, d)
        final = self.cross_attn(final)  # (B, d) — cross-asset comparison

        return self.output_fc(final).squeeze(-1)  # (B,)

    def get_feature_importance(self):
        """Return average VSN weights from last forward pass."""
        if self._last_vsn_weights is None:
            return None
        return self._last_vsn_weights.detach().mean(dim=(0, 1)).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════

def build_model(model_type, n_features, n_assets, cfg):
    """Create model from config."""
    if model_type == 'lstm':
        return LSTMBaseline(
            n_features=n_features,
            n_assets=n_assets,
            hidden_dim=cfg.get('hidden_dim', 32),
            n_layers=cfg.get('lstm_layers', 1),
            dropout=cfg.get('dropout', 0.30),
        )
    elif model_type == 'tft':
        return TemporalFusionTransformer(
            n_features=n_features,
            n_assets=n_assets,
            d_model=cfg.get('hidden_dim', 32),
            n_heads=cfg.get('num_heads', 2),
            lstm_layers=cfg.get('lstm_layers', 1),
            dropout=cfg.get('dropout', 0.30),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
