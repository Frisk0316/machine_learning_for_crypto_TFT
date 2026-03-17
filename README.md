# Machine Learning for Crypto — LSTM & Temporal Fusion Transformer

> **v5.2 實驗：修正時間序列模型的橫截面排序缺陷**
>
> 基於 v4 發現的四大缺陷（範式錯配、過度參數化、損失函數錯誤、時序不穩定），
> 本版進行系統性改善：ListNet 排序損失、Cross-Asset Attention、降低模型容量、
> 時序位置編碼，並與 v3 Gated FFN / v4 原始結果對照。
>
> **v5.2 更新**：32 seeds（原 8）、CUDA 確定性設定、+Trump 特徵組採用 Trump-aware 訓練起點。

---

## 1. Motivation / 研究動機

### 1.1 v4 診斷出的四大缺陷

v4 實驗（原始 LSTM / TFT）揭示了時間序列模型在橫截面排序任務上的系統性劣勢：

| # | 缺陷 | v4 表現 |
|---|------|--------|
| 1 | **任務本質錯配**：MSE 損失最小化絕對預測誤差，但任務需要相對排序 | TFT SR=+0.08, LSTM SR=+1.08 (均遠低於 Gated FFN SR=+2.90) |
| 2 | **無跨資產交互**：每個資產獨立處理，無法比較不同資產 | Decile 排序幾乎隨機（TFT 無單調模式） |
| 3 | **過度參數化**：TFT 368K 參數 / 14K 樣本 = 25.4 比值 | 驗證→測試 Sharpe 平均衰退 -1.0 至 -2.0 |
| 4 | **時序模式不穩定**：12 週回看窗口在加密市場中不穩定 | 高驗證 SR 無法預測高測試 SR |

### 1.2 v5 改善策略

針對每個缺陷提出對應改善：

| 缺陷 | v4 做法 | v5 改善 |
|------|---------|---------|
| 損失函數錯誤 | MSE（最小化絕對誤差） | **ListNet Ranking Loss**（直接優化排序） |
| 無跨資產交互 | 獨立處理每個資產 | **Cross-Asset Self-Attention** |
| 過度參數化 | TFT 368K, LSTM 82K | 降至 TFT **~31K**, LSTM **~18K** |
| 時序不穩定 | lookback=12 | lookback=8, 加 **position embedding** |
| 正則化不足 | dropout=0.15, decay=1e-5 | dropout=**0.30**, decay=**1e-4** |
| 訓練不穩定 | 直接 lr=0.001 | **Linear warmup** 10 epochs |

---

## 2. Architecture / 模型架構

### 2.1 LSTM v5 (with Cross-Asset Attention)

```
Input (N, L=8, M) → Input Projection (M → d=32)
    + Temporal Position Embedding (8 positions, learnable)
    → 1-layer LSTM (d=32)
    → Last hidden state (N, d)
    ⊕ Asset Embedding (86 → d=32)
    → Cross-Asset Self-Attention (2 heads)  ← NEW
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~18K** (v4: ~82K, 4.6× reduction)
- Cross-Asset Attention allows direct comparison between assets

### 2.2 TFT v5 (with Cross-Asset Attention + LightVSN)

```
Input (N, L=8, M) → Per-variable embedding (M × (weight, bias) → d=32)
    + Temporal Position Embedding (8 positions, learnable)
    → Light Variable Selection Network (LightVSN)
        ├─ Shared transform for per-variable processing
        ├─ Pooled + context → softmax weights (interpretable)
        └─ Static context: Asset Embedding (86 → d=32)
    → 1-layer LSTM encoder (d=32)
    → Gated skip connection + LayerNorm
    → Interpretable Multi-Head Attention (2 heads, shared V)
    → Gated skip connection + LayerNorm
    → Cross-Asset Self-Attention (2 heads)  ← NEW
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~29-31K** (v4: ~209-368K, 7-12× reduction)
- LightVSN: ~5K params vs v4 VSN ~55K params (for 33 features)
- Cross-Asset Attention: enables cross-sectional comparison after temporal encoding

### 2.3 v4 → v5 Architecture Comparison

| Component | v4 | v5 | Rationale |
|-----------|----|----|-----------|
| LSTM layers | 2 | **1** | Reduce capacity |
| Hidden dim | 64 | **32** | Reduce capacity |
| Attention heads | 4 | **2** | Match smaller dim |
| VSN | Full GRN (M×d flattened) | **LightVSN** (pooled) | 11× fewer params |
| Cross-asset | None | **Self-Attention** | Enable ranking |
| Position encoding | None | **Learnable** | Time step awareness |
| Output head | FC(2H→H→1) or GRN→Linear | **Linear(d→1)** | Ranking only needs scores |

### 2.4 Cross-Asset Self-Attention (Key Innovation)

```python
class CrossAssetAttention:
    """
    After temporal encoding, each asset has representation h_i ∈ R^d.
    Stack all N assets: H = [h_1, ..., h_N] ∈ R^(N, d)
    Apply self-attention: H' = LayerNorm(H + MHA(H, H, H))

    This lets asset i attend to ALL other assets at the same time step,
    enabling direct cross-sectional comparison for ranking.
    """
```

v4 的根本問題在於每個資產獨立通過 LSTM/TFT，模型無法比較不同資產的表現。Cross-Asset Attention 在時序編碼後加入一層跨資產自注意力，讓每個資產的表示可以參考所有其他資產，從而實現橫截面排序。

### 2.5 ListNet Ranking Loss (Key Innovation)

```python
def listnet_loss(pred, target, mask, temperature=1.0):
    """
    Cross-entropy between softmax distributions:
    L = -Σ softmax(target/τ) · log_softmax(pred/τ)

    Directly optimises the model to produce correct relative orderings,
    not absolute return predictions.
    """
```

v4 使用 MSE 損失，最小化絕對預測誤差。但投資組合構建只需要正確的相對排序。ListNet 將真實報酬和預測分數都轉換為概率分佈（softmax），然後用交叉熵衡量排序相似度。這直接對齊了訓練目標與評估指標。

---

## 3. Data / 數據

**Source**: `deep_learning_for_crypto/datasets/btc_panel.npz` (shared with v3/v4)

| Dimension | Value |
|-----------|-------|
| Time (T) | 324 weeks (2020-01-05 → 2026-03-15) |
| Assets (N) | 86 cryptocurrencies (top 100 excl. stablecoins) |
| Features (M) | 49 (used up to 38 in experiments) |
| Target | Next-week cross-sectional return |
| Missing data | UNK = −99.99, masked in loss and evaluation |

**Feature categories:**
- **Price Momentum (0–4):** r1w, r4w, r12w, r26w, r52w
- **Technical (5–10):** RSI, Bollinger, vol_ratio, ATR, OBV, vol_usd
- **On-chain (11–15):** active_addr, tx_count, NVT, exchange_net_flow, MVRV
- **Macro/Sentiment (16–26):** Fear & Greed, S&P500, DXY, VIX, gold, silver, DJI
- **ETF + Polymarket (27–32):** BTC/ETH ETF flows, Polymarket BTC probability, ETF volume
- **Trump Social Media (44–48):** trump_post_count, trump_caps_ratio, trump_tariff_score, trump_crypto_score, trump_sentiment

### 3.1 Trump Social Media Features (v5.1 新增)

從 [trump-code](https://github.com/sstklen/trump-code) 專案提取 5 個週頻宏觀特徵，捕捉 Trump 社群媒體行為對加密市場的影響：

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 44 | `trump_post_count` | Truth Social + X | 每週發文數（silence = bullish signal） |
| 45 | `trump_caps_ratio` | 文本分析 | 平均大寫字母比例（情緒強度指標） |
| 46 | `trump_tariff_score` | 關鍵詞匹配 | 關稅/貿易戰相關貼文比例 |
| 47 | `trump_crypto_score` | 關鍵詞匹配 | 加密貨幣相關貼文比例 |
| 48 | `trump_sentiment` | 標點分析 | 感嘆號密度（情緒代理） |

**資料來源**：
- Truth Social archive（14,623 篇原創貼文，2022-02 → 2024-06）
- X/Twitter archive（168 篇推文，2025-01 → 2026-03）

**覆蓋率**：209/324 週有效（64.5%），2020-01 至 2022-02 為 UNK（Trump 尚未使用 Truth Social）

**標準化**：與其他宏觀特徵相同，使用 52 週滾動 z-score

**Chronological split (70 / 15 / 15):**
- Train: weeks 0–226 (227 weeks) — 預設
- Train (+Trump): weeks ~115–226 (~112 weeks) — 從 Trump 資料覆蓋開始，自動偵測
- Valid: weeks 227–274 (48 weeks)
- Test: weeks 275–323 (49 weeks)

**v5 Sequence construction (vs v4):**

| | v4 | v5 |
|-|----|----|
| Lookback | L = 12 weeks | **L = 8 weeks** |
| Training batch | Random (asset, time) pairs, batch=128 | **Cross-sectional**: all 86 assets at one time step |
| Training samples | ~14,458 individual pairs | **~218 time steps** (each with N=86 assets) |
| Loss computation | Per-sample MSE | **Per-time-step ListNet** (ranks all 86 assets) |

---

## 4. Quick Start / 快速開始

### Prerequisites

```bash
pip install torch numpy matplotlib scipy
```

### One-click execution (全部一鍵執行)

```bash
cd 論文/machine_learning_for_crypto_TFT
bash run_all.sh
```

### Step-by-step

```bash
# Step 1: Train TFT v5 (32 seeds × 4 info sets, ~54 min on GPU)
python train.py --config config.json --model tft

# Step 2: Train LSTM v5 (32 seeds × 4 info sets, ~34 min on GPU)
python train.py --config config.json --model lstm

# Step 3: Evaluate TFT
python evaluate.py --config config.json --model tft

# Step 4: Evaluate LSTM
python evaluate.py --config config.json --model lstm
```

### Outputs

```
checkpoints/
├── tft_results.npz      # TFT ensemble predictions
└── lstm_results.npz     # LSTM ensemble predictions

outputs/
├── tft_table3.csv       # TFT performance summary
├── tft_cumulative_returns.png
├── tft_decile_sharpe.png
├── tft_feature_importance.png
├── lstm_table3.csv      # LSTM performance summary
├── lstm_cumulative_returns.png
├── lstm_decile_sharpe.png
└── lstm_feature_importance.png
```

---

## 5. Hyperparameters / 超參數

| Parameter | v4 | **v5** | Change Reason |
|-----------|----|----|---------------|
| `lookback` | 12 | **8** | 減少時序雜訊 |
| `hidden_dim` | 64 | **32** | 降低過擬合 |
| `num_heads` | 4 | **2** | 匹配更小維度 |
| `lstm_layers` | 2 | **1** | 降低容量 |
| `dropout` | 0.15 | **0.30** | 加強正則化 |
| `weight_decay` | 1e-5 | **1e-4** | 加強 L2 |
| `learning_rate` | 0.001 | 0.001 | 保持不變 |
| `batch_size` | 128 | **N/A** (cross-sectional) | 每個時間步即一個 batch |
| `epochs` | 150 | 150 | 保持不變 |
| `early_stopping_patience` | 20 | 20 | 保持不變 |
| `warmup_epochs` | N/A | **10** | 穩定早期訓練 |
| `ranking_temperature` | N/A | **1.0** | ListNet softmax 溫度 |
| `num_seeds` | 8 | **32** | 增加至 32 以降低 ensemble 變異數 |

---

## 6. Results / 實驗結果

### 6.1 Table 3: Long-Short Portfolio Performance (Test Period, 48 weeks)

**TFT v5.2 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | mean PW% | t-stat | SR (EW) | Ensemble SR | Mean±Std |
|-----------------|:-------:|:--------:|:------:|:-------:|:-----------:|:--------:|
| Price+Technical (11) | +0.08 | +0.12 | +0.07 | -0.20 | -0.198 | -0.395±0.406 |
| +Onchain (16) | -0.39 | -0.55 | -0.37 | -0.64 | -0.645 | -0.526±0.512 |
| All (33) | -0.03 | -0.05 | -0.03 | +0.04 | +0.038 | -0.172±0.421 |
| **+Trump (38)** | **+0.15** | +0.20 | +0.15 | **+0.29** | **+0.294** | -0.065±0.605 |

**LSTM v5.2 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | mean PW% | t-stat | SR (EW) | Ensemble SR | Mean±Std |
|-----------------|:-------:|:--------:|:------:|:-------:|:-----------:|:--------:|
| **Price+Technical (11)** | **+0.89** | +1.25 | +0.85 | +0.57 | +0.567 | -0.020±0.608 |
| +Onchain (16) | +0.63 | +0.92 | +0.60 | +0.41 | +0.407 | +0.002±0.454 |
| All (33) | -0.45 | -0.43 | -0.43 | -0.07 | -0.070 | -0.044±0.474 |
| **+Trump (38)** | **+0.44** | +0.65 | +0.42 | +0.22 | +0.216 | +0.606±0.757 |

### 6.2 v4 → v5.2 Improvement / v4 vs v5.2 改善比較

**TFT:**

| Information Set | v4 SR (PW) | **v5.2 SR (PW)** | Δ SR | Improved? |
|-----------------|-----------|---------------|------|-----------|
| Price+Technical | −0.04 | **+0.08** | +0.12 | ✓ |
| +Onchain | −0.89 | −0.39 | +0.50 | ✓ |
| All | +0.08 | −0.03 | −0.11 | ≈ |

**LSTM:**

| Information Set | v4 SR (PW) | **v5.2 SR (PW)** | Δ SR | Improved? |
|-----------------|-----------|---------------|------|-----------|
| Price+Technical | −1.25 | **+0.89** | **+2.14** | ✓✓ |
| +Onchain | −1.27 | **+0.63** | **+1.90** | ✓✓ |
| All | +1.08 | −0.45 | −1.53 | ✗ |

**LSTM v5.2 在 Price+Technical 和 +Onchain 上實現了最大的改善**（分別 +2.14 和 +1.90 SR），從顯著為負翻轉為正值。32 seeds 的 ensemble 結果比 v5.1 的 8 seeds 更穩定可靠。

### 6.3 Full Cross-Model Comparison / 全模型比較 (SR PW, Long-Short)

| Information Set | Gated FFN (v3) | TFT v4 | **TFT v5.2** | LSTM v4 | **LSTM v5.2** |
|-----------------|----------------|--------|:-----------:|---------|:------------:|
| **Price+Technical** | **+2.10** (t=2.04) ✓ | −0.04 | +0.08 | −1.25 | **+0.89** |
| **+Onchain** | −0.95 | −0.89 | −0.39 | −1.27 | **+0.63** |
| **All features** | **+2.90** (t=2.82) ✓✓ | +0.08 | −0.03 | +1.08 | −0.45 |
| **+Trump** | N/A | N/A | +0.15 | N/A | **+0.44** |

> **LSTM v5.2 Price+Technical (SR=+0.89) 是所有時間序列模型中的最佳表現。**

### 6.4 Decile Analysis / 十分位分析

**LSTM v5.2 — Price+Technical (best config, SR=+0.89):**
| Decile | Top | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | Bot | L−S |
|--------|-----|----|----|----|----|----|----|----|----|-----|-----|
| SR (PW) | +0.18 | −1.00 | −0.69 | −1.36 | −0.58 | −0.73 | −0.50 | +0.49 | −0.54 | −0.67 | +0.85 |

**LSTM v5.2 — +Trump (SR=+0.44):**
| Decile | Top | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | Bot | L−S |
|--------|-----|----|----|----|----|----|----|----|----|-----|-----|
| SR (PW) | −0.03 | −0.62 | −0.13 | −0.64 | −1.10 | −0.65 | −0.75 | −0.33 | −1.19 | −0.75 | +0.72 |

**TFT v5.2 — +Trump (SR=+0.15):**
| Decile | Top | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | Bot | L−S |
|--------|-----|----|----|----|----|----|----|----|----|-----|-----|
| SR (PW) | −0.11 | −0.72 | −0.03 | −1.24 | −0.90 | −0.97 | −0.87 | −0.82 | +0.17 | −0.47 | +0.36 |

**Gated FFN (v3) — All features (for comparison):**
| Decile | Top | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | Bot | L−S |
|--------|-----|----|----|----|----|----|----|----|----|-----|-----|
| SR (PW) | **+0.84** | — | — | — | — | — | — | — | — | **−1.95** | **+2.79** |

LSTM v5.2 Price+Technical 的 L−S=+0.85 是時間序列模型中最佳，但仍遠不及 Gated FFN 的清晰單調排序（2.79）。

### 6.5 Training Efficiency / 訓練效率

| Model | v4 Time (CPU) | v5 Time (CPU) | **v5.2 Time (GPU, 32 seeds)** | v4 Params | v5 Params | Reduction |
|-------|---------|---------|---------|-----------|-----------|-----------|
| TFT | 120.6 min | 62.3 min (8s) | **53.8 min** | 209-368K | **29-32K** | 7-12× |
| LSTM | 16.9 min | 9.0 min (8s) | **33.9 min** | 81-83K | **18-19K** | 4.6× |

v5.2 使用 32 seeds × 4 feature configs = 128 次訓練。GPU（RTX 3060 Ti）上 TFT 約 54 分鐘、LSTM 約 34 分鐘完成。CUDA 確定性設定確保完全可重現。

### 6.6 TFT Variable Selection Importance / TFT 特徵重要性（LightVSN 權重）

TFT v5 的 LightVSN 仍提供可解釋的特徵重要性權重（平均 softmax 權重）。由於使用了更輕量的架構和排序損失，特徵權重分佈與 v4 可能有所不同。

---

## 7. Conclusion / 結論

### 7.1 v5.2 改善的效果：部分成功

v5 的三項核心改善（ListNet + Cross-Asset Attention + 降低容量）確實改善了部分配置的績效。v5.2 使用 32 seeds + CUDA 確定性設定後，結果更加穩定可靠：

| 改善 | 效果 |
|------|------|
| **LSTM Price+Technical** | SR: −1.25 → **+0.89** (✓✓ 最大改善) |
| **LSTM +Onchain** | SR: −1.27 → **+0.63** (✓✓ 從負轉正) |
| **LSTM +Trump** | **+0.44** (✓ 新特徵組，正向貢獻) |
| TFT Price+Technical | SR: −0.04 → +0.08 (✓ 小幅改善) |
| TFT +Onchain | SR: −0.89 → −0.39 (✓ 改善) |
| TFT +Trump | **+0.15** (✓ 新特徵組，正向貢獻) |
| TFT All | SR: +0.08 → −0.03 (≈ 持平) |
| LSTM All | SR: +1.08 → −0.45 (✗ 惡化) |

### 7.2 為何 LSTM 改善幅度大於 TFT

LSTM v5.2 在 Price+Technical 上從 SR=−1.25 躍升至 SR=+0.89，改善幅度是所有配置中最大的。原因：

1. **LSTM 更受惠於 Cross-Asset Attention**：LSTM 的原始架構極為簡單（僅 recurrent memory），加入 Cross-Asset Attention 後等同於為其增加了全新的跨資產比較能力
2. **LSTM 更受惠於 Ranking Loss**：簡單模型在 MSE 下容易過擬合絕對值，但在 Ranking Loss 下只需學習相對順序，降低了學習難度
3. **較少參數更不易過擬合**：LSTM v5 僅 18K 參數，在 218 個 cross-sectional batches 上相對不易過擬合

### 7.3 為何 All features 惡化

LSTM v5.2 All features 從 SR=+1.08 降至 −0.45，原因可能是：

1. **33 特徵 × 86 資產 × 8 時間步**的組合使 Cross-Asset Attention 的輸入更加嘈雜
2. **ListNet 在特徵過多時更敏感**：更多特徵意味著模型需要在更大的特徵空間中學習排序，而 ListNet 的 softmax 分佈對噪聲特徵更敏感
3. **v4 LSTM All features 的 +1.08 可能是隨機性**：v4 的 per-seed std=0.711 很大，+1.08 可能只是噪聲中的幸運結果

### 7.4 Trump 特徵的增量價值 (v5.2)

加入 5 個 Trump 社群媒體特徵後，兩個模型均獲得改善：

| Model | All (33) SR PW | +Trump (38) SR PW | Δ SR |
|-------|:--:|:--:|:--:|
| LSTM | -0.45 | **+0.44** | **+0.89** |
| TFT | -0.03 | **+0.15** | **+0.18** |

**為何 Trump 特徵有效：**

1. **政策不確定性定價**：Trump 的關稅推文（`tariff_score`）直接影響全球風險偏好，加密市場對此尤為敏感
2. **直接因果通道**：2025 年 Trump 簽署加密行政命令、提出 Strategic Bitcoin Reserve，`crypto_score` 捕捉了這一直接影響
3. **情緒互補性**：Trump 推文情緒（`caps_ratio`, `sentiment`）與 Fear & Greed Index 提供不同維度的市場情緒信號
4. **沉默信號**：`post_count = 0` 的週代表無政策衝擊，trump-code 驗證 80% 的沉默日後市上漲

**v5.2 Trump-aware 訓練**：+Trump 特徵組的訓練起點自動從 Trump 資料覆蓋開始（week 114, 2022-03-13），所有訓練樣本皆有有效 Trump 特徵。訓練週數為 ~112 週（~104 cross-sectional batches），消除了 UNK 填充造成的噪聲。

**限制：**
- 覆蓋率僅 64.5%（209/324 週），2020-2022 期間完全缺失
- +Trump 訓練集較小（104 vs 218 batches），但 32 seeds ensemble 緩解了高變異問題
- 因果方向不明確：Trump 可能是對市場的反應而非驅動因素

### 7.5 核心結論：Gated FFN 仍然是最佳選擇

即便加入 Trump 特徵後有所改善，Gated FFN (v3) 仍以巨大優勢領先：

| Criterion | Gated FFN (v3) | TFT v5.2 best | LSTM v5.2 best |
|-----------|----------------|:------:|:------:|
| **Best SR (PW)** | **+2.90** ✓✓ | +0.15 (+Trump) | **+0.89** (Price+Tech) |
| **Decile L−S** | **+2.79** ✓✓ | +0.36 | +0.85 |
| **Training time (GPU)** | ~30 min | 54 min (32s) | 34 min (32s) |
| **Parameter count** | **~4K** ✓✓ | ~32K | ~19K |

**v5 的改善證實了我們診斷的部分缺陷確實是真實的**（尤其是 Ranking Loss 和 Cross-Asset Attention 對 LSTM 的幫助），**但無法彌合橫截面模型的根本優勢**：

> 對於週頻加密貨幣橫截面報酬排序，純橫截面模型（Gated FFN, 4K 參數）仍遠優於經過充分改善的時間序列模型（LSTM v5, 18K 參數）。時間序列建模提供的歷史記憶無法彌補「同時看到所有資產」的橫截面優勢。

### 7.6 改善的價值：診斷性洞見

儘管 v5 未能超越 Gated FFN，改善過程提供了有價值的診斷性洞見：

1. **Ranking Loss 有效**：將訓練目標對齊評估指標，確實改善了排序品質（LSTM Price+Tech: −1.25 → +0.89）
2. **Cross-Asset Attention 有效**：讓獨立處理的時間序列模型獲得跨資產比較能力，改善了排序
3. **降低容量有效**：18K 參數的 LSTM v5 優於 82K 參數的 LSTM v4，驗證了過擬合是 v4 的主要問題
4. **但根本限制仍在**：即便有排序損失和跨資產注意力，時序模型仍需通過「先獨立編碼，後跨資產比較」的兩步流程，而橫截面模型天然地在一步中完成比較

---

## 8. Limitations / 研究限制

1. **Short test period**: 49 weeks of out-of-sample evaluation
2. **Single lookback**: Only L=8 tested; {4, 12, 24} may yield different results
3. **ListNet temperature**: Only τ=1.0 tested; different temperatures may change ranking sharpness
4. **Cross-Asset Attention is simple**: Only 1 layer of self-attention; deeper cross-asset modeling (e.g., Transformer blocks) may improve ranking
5. **No hybrid approach**: Could combine ListNet on cross-sectional Gated FFN with temporal features, potentially getting best of both worlds
6. **Trump data coverage gap**: Trump 特徵僅覆蓋 64.5% 的週次（2022-03 起）。v5.2 透過 Trump-aware 訓練起點緩解了此問題，但訓練集較小（104 vs 218 batches）
7. **Trump causality unclear**: Trump 推文可能是對市場的反應（lag）而非市場的驅動因素（lead），需要更嚴格的 Granger 因果檢驗

---

## 9. Future Work / 待研究方向

1. **Hybrid Cross-Sectional + Temporal**: 使用 LSTM/TFT 作為特徵提取器（per-asset temporal encoding），然後將編碼結果輸入 Gated FFN 進行橫截面排序。這結合了時序記憶和橫截面比較的優勢
2. **ListNet on Gated FFN**: 將 ListNet 排序損失應用於 v3 Gated FFN，可能進一步提升其 SR=2.90 的結果
3. **Trump features on Gated FFN**: 將 Trump 社群媒體特徵加入 Gated FFN，驗證其在橫截面模型中的增量價值
4. **Deeper Cross-Asset Modeling**: 多層 Transformer 編碼器取代單層 Cross-Asset Attention
5. **Per-feature lookback tuning**: 不同特徵類別使用不同的 lookback 窗口（e.g., momentum 用 L=4, macro 用 L=12）
6. **Temperature annealing**: ListNet 溫度從高到低退火，先學粗略排序再學精細排序
7. **Trump signal refinement**: 使用 trump-code 的 384 維日頻特徵（`daily_features.json`）替代我們手工的 5 維週頻特徵，或引入 NLP 模型（sentiment analysis）提取更精確的情緒分數
8. **Granger causality test**: 對 Trump 特徵進行 Granger 因果檢驗，確認其對加密報酬的預測方向性

---

## 10. v4 → v5.2 Changes Summary / 改版摘要

| Component | v4 | v5.2 |
|-----------|----|----|
| **Loss function** | MSE | **ListNet ranking loss** |
| **Training paradigm** | Random (asset, time) pairs | **Cross-sectional batches** |
| **Cross-asset** | None | **Self-Attention (2 heads)** |
| **VSN** | Full GRN (55K params for M=33) | **LightVSN (5K params)** |
| **Hidden dim** | 64 | **32** |
| **LSTM layers** | 2 | **1** |
| **Dropout** | 0.15 | **0.30** |
| **Weight decay** | 1e-5 | **1e-4** |
| **Lookback** | 12 | **8** |
| **Position encoding** | None | **Learnable** |
| **LR warmup** | None | **10 epochs linear** |
| **Num seeds** | 8 | **32** |
| **Deterministic** | No | **Yes (CUDA deterministic)** |
| **Trump-aware train** | N/A | **Auto-detect start week** |
| **TFT params** | 209-368K | **29-31K** |
| **LSTM params** | 81-83K | **18K** |
| **TFT train time** | 120 min (CPU, 8s) | **53.8 min (GPU, 32s)** |
| **LSTM train time** | 17 min (CPU, 8s) | **33.9 min (GPU, 32s)** |

---

## 11. File Structure / 檔案結構

```
machine_learning_for_crypto_TFT/
├── README.md                 # This file
├── config.json               # v5 hyperparameters and feature configs
├── data_loader.py            # Load btc_panel.npz → cross-sectional batches
├── models.py                 # LSTM + TFT with Cross-Asset Attention
├── train.py                  # ListNet ranking loss training loop
├── evaluate.py               # Portfolio evaluation + visualization
├── run_all.sh                # One-click execution
├── checkpoints/
│   ├── tft_results.npz       # TFT ensemble predictions
│   └── lstm_results.npz      # LSTM ensemble predictions
└── outputs/
    ├── tft_table3.csv
    ├── tft_cumulative_returns.png
    ├── tft_decile_sharpe.png
    ├── tft_feature_importance.png
    ├── lstm_table3.csv
    ├── lstm_cumulative_returns.png
    ├── lstm_decile_sharpe.png
    └── lstm_feature_importance.png
```

---

## 12. References

1. Cao, Z., Qin, T., Liu, T. Y., Tsai, M. F., & Li, H. (2007). **Learning to Rank: From Pairwise Approach to Listwise Approach**. *ICML*. — ListNet ranking loss
2. Lim, B., Arık, S. Ö., Loeff, N., & Pfister, T. (2021). **Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting**. *International Journal of Forecasting*, 37(4), 1748–1764.
3. Vaswani, A., et al. (2017). **Attention Is All You Need**. *NeurIPS*. — Cross-asset self-attention
4. Hochreiter, S., & Schmidhuber, J. (1997). **Long Short-Term Memory**. *Neural Computation*, 9(8), 1735–1780.
5. Research Project Literature Survey: 743 papers (2016–2026) on cryptocurrency × machine learning. See `論文/research_project/data/research_gap_report.md`.
6. **Trump Code** — AI-powered cryptanalysis of presidential communications × stock market impact. [github.com/sstklen/trump-code](https://github.com/sstklen/trump-code). 31.5M model combinations tested, 551 surviving rules, 61.3% hit rate (z=5.39). Source of Trump social media features (v5.1).
