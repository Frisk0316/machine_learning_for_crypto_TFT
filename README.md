# Machine Learning for Crypto — LSTM & Temporal Fusion Transformer

> **v5.4 實驗：時間序列模型 vs 傳統模型 vs 橫截面模型**
>
> 基於 v4 發現的四大缺陷（範式錯配、過度參數化、損失函數錯誤、時序不穩定），
> 本版進行系統性改善：ListNet 排序損失、Cross-Asset Attention、適度模型容量、
> 時序位置編碼，並加入傳統 ML 基準（OLS, ElasticNet, PCA, PLS, RF, GBT）和
> VW/EW 市場組合基準，進行全面的模型比較。
>
> **v5.4 變更**：
> - hidden_dim 32 → **64**（增加模型容量，提升 LSTM/TFT 表現潛力）
> - RF/GBT 新增 **5-seed ensemble**（與 DL 32-seed ensemble 公平比較）
> - 市場基準改用 **VW Market Portfolio**（市值加權，CoinGecko 實際市值資料）
> - EW Market 降為輔助參考
> - 資料已在 `prepare_btc_data.py` 做好橫截面排名正規化，無需額外處理

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
Input (N, L=8, M) → Input Projection (M → d=64)
    + Temporal Position Embedding (8 positions, learnable)
    → 1-layer LSTM (d=64)
    → Last hidden state (N, d)
    ⊕ Asset Embedding (86 → d=64)
    → Cross-Asset Self-Attention (2 heads)  ← NEW
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~53K** (v4: ~82K, v5.3: ~18K)
- Cross-Asset Attention allows direct comparison between assets

### 2.2 TFT v5 (with Cross-Asset Attention + LightVSN)

```
Input (N, L=8, M) → Per-variable embedding (M × (weight, bias) → d=64)
    + Temporal Position Embedding (8 positions, learnable)
    → Light Variable Selection Network (LightVSN)
        ├─ Shared transform for per-variable processing
        ├─ Pooled + context → softmax weights (interpretable)
        └─ Static context: Asset Embedding (86 → d=64)
    → 1-layer LSTM encoder (d=64)
    → Gated skip connection + LayerNorm
    → Interpretable Multi-Head Attention (2 heads, shared V)
    → Gated skip connection + LayerNorm
    → Cross-Asset Self-Attention (2 heads)  ← NEW
    → LayerNorm + residual
    → Linear(d → 1)
    → Ranking scores (N,)
```

- **Parameters: ~90-100K** (v4: ~209-368K, v5.3: ~29-31K)
- LightVSN: ~12K params vs v4 VSN ~55K params (for 33 features)
- Cross-Asset Attention: enables cross-sectional comparison after temporal encoding

### 2.3 v4 → v5 Architecture Comparison

| Component | v4 | v5 | Rationale |
|-----------|----|----|-----------|
| LSTM layers | 2 | **1** | Reduce capacity |
| Hidden dim | 64 | **64** | v5.4 恢復為 64（v5.3 為 32 過於保守） |
| Attention heads | 4 | **2** | Fewer heads sufficient |
| VSN | Full GRN (M×d flattened) | **LightVSN** (pooled) | Fewer params |
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

### 3.2 Look-Ahead Bias 分析

Trump 社群媒體特徵（indices 44–48）**不存在 look-ahead bias**。以下逐步驗證：

#### 3.2.1 原始特徵計算（`fetch_trump.py`）

```python
week_posts = [p for p in parsed if week_start < p["dt"] <= week_end]
```

每週的 5 個特徵（post_count, caps_ratio, tariff_score, crypto_score, sentiment）僅使用 `(dates[t-1], dates[t]]` 區間內的貼文計算。**不使用未來資料**。

#### 3.2.2 滾動 z-score 標準化（`prepare_btc_data.py`）

```python
roll_mean = ser.rolling(52, min_periods=4).mean()  # window = [t-51, ..., t]
roll_std  = ser.rolling(52, min_periods=4).std()
z[t] = (col[t] - roll_mean[t]) / roll_std[t]
```

`rolling(52)` 在時間 t 使用 `[t-51, ..., t]` 的資料，**僅包含過去和當前值**。當前值包含在自身標準化中（~1/52 = 1.9% 權重），這是金融時間序列的標準做法（如 Gu, Kelly & Xiu 2020 對股票特徵的處理）。

可選改善：加 `.shift(1)` 使標準化更嚴格（僅用 t-1 以前的資料），但實質影響極小。

#### 3.2.3 已知偏誤（非 Trump 特有）

| 偏誤類型 | 描述 | 影響 |
|----------|------|------|
| **存活偏誤** | 86 資產依 2026-03 市值選取 | 所有特徵共同存在，非 Trump 特有 |
| **覆蓋率缺口** | Truth Social: 2022-02→2024-06, X: 2025-01→2026-03 | 中間 ~6 個月為 UNK，正確行為 |

#### 3.2.4 結論

Trump 特徵在特徵計算和標準化兩個環節均**嚴格使用過去和當前資料**，不存在 look-ahead bias。唯一的設計選擇（rolling window 包含當前值）是金融計量學的標準做法。

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
pip install torch numpy matplotlib scipy scikit-learn
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

# Step 3: Train traditional models (OLS, EN, PCA, PLS, RF, GBT, ~4 min)
python train_traditional.py --config config.json

# Step 4: Evaluate individual models
python evaluate.py --config config.json --model tft
python evaluate.py --config config.json --model lstm
python evaluate.py --config config.json --model ols  # etc.

# Step 5: Cross-model comparison table
python evaluate.py --config config.json --compare-all
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
| `hidden_dim` | 64 | **64** | v5.4 恢復（v5.3 為 32 過於保守） |
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

### 6.1 Table 3: Long-Short Portfolio Performance (Test Period)

> **v5.4 結果** — hidden_dim=64, 32 seeds ensemble, ListNet ranking loss
> 訓練完成後將更新此表。

**TFT v5.4 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | mean PW% | t-stat | SR (EW) |
|-----------------|:-------:|:--------:|:------:|:-------:|
| Price+Technical (11) | −1.12 | −1.17% | −1.08 | −0.80 |
| +Onchain (16) | −1.04 | −1.17% | −1.00 | −0.58 |
| All (33) | −0.81 | −1.03% | −0.78 | −0.93 |
| +Trump (38) | **+0.48** | +0.72% | +0.46 | +0.31 |

**LSTM v5.4 Ensemble (32 seeds, deterministic):**

| Information Set | SR (PW) | mean PW% | t-stat | SR (EW) |
|-----------------|:-------:|:--------:|:------:|:-------:|
| Price+Technical (11) | +0.72 | +0.97% | +0.69 | +0.67 |
| +Onchain (16) | +0.47 | +0.68% | +0.45 | +0.25 |
| All (33) | −0.19 | −0.17% | −0.18 | −0.04 |
| +Trump (38) | **+1.08** | +1.34% | +1.04 | +1.39 |

### 6.2 v4 → v5.4 Improvement / v4 vs v5.4 改善比較

| Model | v4 best SR (PW) | v5.4 best SR (PW) | Best Config | Δ SR |
|-------|:---------------:|:-----------------:|:-----------:|:----:|
| TFT | +0.08 | **+0.48** | +Trump | +0.40 |
| LSTM | +1.08 | **+1.08** | +Trump | ±0.00 |

v5.4 的 TFT 在 +Trump 特徵組取得 SR=+0.48（v4 最佳 +0.08，改善 +0.40）。LSTM 在 +Trump 特徵組達到 SR=+1.08，與 v4 持平，但 v4 沒有 +Trump 特徵組，屬於不同條件下的比較。注意：使用 All features 時，LSTM v5.4 SR=−0.19，TFT v5.4 SR=−0.81，均弱於 v4（MSE loss 下的 LSTM All=+1.08，TFT All=+0.08）；Trump 特徵的加入對 DL 模型貢獻顯著（LSTM: +1.27 Δ SR vs All features）。

### 6.3 Full Cross-Model Comparison / 全模型比較 (SR PW, Long-Short)

| Information Set | VW Mkt | EW Mkt | Gated FFN | OLS | EN | PCA | PLS | RF (5s) | GBT (5s) | TFT v5.4 | LSTM v5.4 |
|-----------------|:------:|:------:|:---------:|:---:|:--:|:---:|:---:|:-------:|:--------:|:--------:|:---------:|
| **Price+Technical** | −0.17 | −0.60 | **+2.10** | +1.13 | −0.78 | +0.43 | +0.58 | +1.31 | +1.53 | −1.12 | +0.72 |
| **+Onchain** | −0.17 | −0.60 | −0.95 | +1.00 | −0.78 | +0.46 | +0.87 | +1.54 | +1.43 | −1.04 | +0.47 |
| **All features** | −0.17 | −0.60 | **+2.90** | +1.24 | −0.78 | +0.56 | +0.73 | +1.53 | +1.16 | −0.81 | −0.19 |
| **+Trump** | −0.17 | −0.60 | N/A | +1.31 | +1.03 | +0.41 | +0.82 | +1.58 | **+1.61** | +0.48 | +1.08 |

> RF/GBT 後括號 (5s) 表示 5-seed ensemble。

### 6.4 Decile Analysis / 十分位分析

> 訓練完成後更新。

### 6.5 Training Efficiency / 訓練效率

| Model | v4 Params | **v5.4 Params** | v4 vs v5.4 | Ensemble |
|-------|-----------|-------------|------------|----------|
| TFT | 209-368K | **~90-100K** | 2-4× 減少 | 32 seeds |
| LSTM | 81-83K | **~53K** | 1.5× 減少 | 32 seeds |
| RF | — | — | — | **5 seeds** (v5.4 新增) |
| GBT | — | — | — | **5 seeds** (v5.4 新增) |
| OLS/EN/PCA/PLS | — | — | — | 1 (確定性) |

v5.4 使用 32 seeds × 4 feature configs = 128 次 DL 訓練。CUDA 確定性設定確保完全可重現。
hidden_dim=64 相比 v5.3 的 32 增加了模型容量（TFT: 31K→95K, LSTM: 18K→53K），但仍遠少於 v4。

### 6.6 TFT Variable Selection Importance / TFT 特徵重要性（LightVSN 權重）

TFT v5 的 LightVSN 仍提供可解釋的特徵重要性權重（平均 softmax 權重）。由於使用了更輕量的架構和排序損失，特徵權重分佈與 v4 可能有所不同。

### 6.7 Traditional Model Baselines (v5.3 新增, v5.4 更新)

**Ensemble 策略：**
- OLS / ElasticNet / PCA-OLS / PLS：確定性模型，固定 HP 下無隨機性，單一模型即等價於 ensemble
- **Random Forest / Gradient Boosting：v5.4 新增 5-seed ensemble**（最佳 HP 選定後，用 5 個不同 random_state 訓練並平均預測，與 DL 32-seed ensemble 公平比較）

**Long-Short SR (PW) — Test Period:**

> 訓練完成後更新。RF/GBT 現在使用 5-seed ensemble（v5.4 變更）。

| Information Set | OLS | ElasticNet | PCA | PLS | RF (5s) | GBT (5s) |
|-----------------|:---:|:----------:|:---:|:---:|:-------:|:--------:|
| Price+Technical (11) | +1.13 | −0.78 | +0.43 | +0.58 | +1.31 | +1.53 |
| +Onchain (16) | +1.00 | −0.78 | +0.46 | +0.87 | +1.54 | +1.43 |
| All (33) | +1.24 | −0.78 | +0.56 | +0.73 | +1.53 | +1.16 |
| +Trump (38) | +1.31 | +1.03 | +0.41 | +0.82 | +1.58 | **+1.61** |

### 6.8 Market Portfolio Benchmark (v5.4 更新)

| Portfolio | mean%/week | SR | T | 資料來源 |
|-----------|:----------:|:--:|:-:|---------|
| **VW Market [PRIMARY]** (value-weighted) | −0.15% | **−0.174** | 48 | CoinGecko 實際市值 |
| EW Market [reference] (equal-weighted) | −0.76% | −0.596 | 48 | 等權平均 |

**v5.4 變更**：市場基準從 EW 改為 **VW Market Portfolio**（市值加權）。
- VW 使用 CoinGecko 免費 API 取得的 86 幣種歷史市值（`market_cap.npz`，`days=365`，涵蓋測試期 100% 覆蓋率）
- VW 更準確反映加密市場的實際表現（BTC/ETH 主導），避免等權平均高估小幣影響
- EW 保留作為參考：等權平均在小樣本橫截面研究中仍有學術參考價值

**EW vs VW 於 portfolio construction 的說明**：
長短倉 portfolio 內部使用 **EW**（equal-weight within each decile），這是橫截面報酬預測的學術標準選擇（cf. Gu, Kelly, Xiu 2020）。理由：(1) 避免對大市值資產的集中，(2) 測試模型對全宇宙的排名能力，(3) 不受市值波動影響。市場基準使用 VW，因其更準確反映被動投資者的實際報酬。

---

## 7. Conclusion / 結論

### 7.1 v5.4 改善策略

v5 的核心改善（ListNet + Cross-Asset Attention + 適度容量），加上 v5.4 的 hidden_dim 恢復為 64，旨在讓 DL 模型有足夠容量學習橫截面排序模式。32 seeds + CUDA 確定性設定確保結果穩定可靠。

**v5.4 最終結果：**
- LSTM 最佳：SR=**+1.08**（+Trump 特徵組），ElasticNet 和 PCA 基準之上，但仍低於 OLS (+1.31)、RF (+1.58)、GBT (+1.61)
- TFT 最佳：SR=**+0.48**（+Trump 特徵組），僅略高於市場 VW (-0.17)，在所有模型中墊底
- 增加 hidden_dim 64 相比 v5.3（32）未帶來明顯改善，LSTM All features SR 反從 v4 +1.08 退至 -0.19，顯示容量擴大導致更嚴重過擬合
- Trump 特徵對 DL 模型幫助最大（LSTM +1.27 Δ SR vs All features），遠超傳統模型 (OLS +0.07, GBT +0.45)

### 7.2 LSTM vs TFT 架構差異分析

1. **LSTM 更受惠於 Cross-Asset Attention**：LSTM 的原始架構極為簡單（僅 recurrent memory），加入 Cross-Asset Attention 後等同於為其增加了全新的跨資產比較能力。實際結果：LSTM 最佳 SR=+1.08，TFT 最佳 SR=+0.48，差距顯著
2. **LSTM 更受惠於 Ranking Loss**：簡單模型在 MSE 下容易過擬合絕對值，但在 Ranking Loss 下只需學習相對順序
3. **v5.4 hidden_dim=64 的影響**：LSTM ~53K 參數（vs v5.3 的 18K），TFT ~95K 參數（vs v5.3 的 31K）。更大容量在 +Trump 特徵組有效（訓練集較小，過擬合壓力較低），但在 All features 下導致 LSTM SR 退至 −0.19，TFT 退至 −0.81，反不如 v4
4. **TFT 的複雜度是負擔**：TFT 有 LightVSN + LSTM + Interpretable Attention + Cross-Asset Attention 四層，在週頻資料（218 個訓練時間步）下嚴重過擬合，幾乎所有特徵組均為負 SR。LSTM 架構更簡單，過擬合程度較輕，在非 All-features 情況下維持正 SR

### 7.3 Trump 特徵的增量價值

**Trump 特徵增量價值（vs All features 特徵組）：**

| Model | All SR (PW) | +Trump SR (PW) | Δ SR |
|-------|:-----------:|:--------------:|:----:|
| LSTM | −0.19 | **+1.08** | **+1.27** |
| TFT | −0.81 | +0.48 | +1.29 |
| GBT | +1.16 | +1.61 | +0.45 |
| OLS | +1.24 | +1.31 | +0.07 |
| RF | +1.53 | +1.58 | +0.05 |
| ElasticNet | −0.78 | +1.03 | **+1.81** |

DL 模型（LSTM, TFT）和 ElasticNet 從 Trump 特徵獲益最大，可能因為 Trump 特徵是相對乾淨的信號（5 維），不會被 All features（33 維）中的雜訊所干擾。線性模型（EN）在 All features 下難以處理多重共線性，Trump 特徵組訓練集較小也導致正則化更有效。

**為何 Trump 特徵有效：**

1. **政策不確定性定價**：Trump 的關稅推文（`tariff_score`）直接影響全球風險偏好，加密市場對此尤為敏感
2. **直接因果通道**：2025 年 Trump 簽署加密行政命令、提出 Strategic Bitcoin Reserve，`crypto_score` 捕捉了這一直接影響
3. **情緒互補性**：Trump 推文情緒（`caps_ratio`, `sentiment`）與 Fear & Greed Index 提供不同維度的市場情緒信號
4. **沉默信號**：`post_count = 0` 的週代表無政策衝擊，trump-code 驗證 80% 的沉默日後市上漲

**v5.4 Trump-aware 訓練**：+Trump 特徵組的訓練起點自動從 Trump 資料覆蓋開始（week 114, 2022-03-13），所有訓練樣本皆有有效 Trump 特徵。

**限制：**
- 覆蓋率僅 64.5%（209/324 週），2020-2022 期間完全缺失
- +Trump 訓練集較小（104 vs 218 batches），但 32 seeds ensemble 緩解了高變異問題
- 因果方向不明確：Trump 可能是對市場的反應而非驅動因素

### 7.5 傳統模型基準：簡單模型的啟示（v5.3 新增，v5.4 確認）

v5.3/v5.4 加入的傳統模型基準揭示了一個一致的重要發現：**即便是最簡單的 OLS，也全面優於 LSTM/TFT 時間序列模型**。v5.4（hidden_dim=64，32 seeds）未能改變此結論，LSTM 最佳 SR=+1.08 仍低於 OLS +1.31，TFT 最佳 SR=+0.48 更只有 OLS 的三分之一。

| Rank | Model | Best SR (PW) | Best Config | Params | Ensemble | Temporal? |
|------|-------|:---:|-------------|:------:|:--------:|:---------:|
| 1 | **Gated FFN** (v3) | **+2.90** | All | ~4K | — | ✗ |
| 2 | **GBT** | **+1.61** | +Trump | — | 5 seeds | ✗ |
| 3 | **RF** | **+1.58** | +Trump | — | 5 seeds | ✗ |
| 4 | **OLS** | **+1.31** | +Trump | — | 1 (deterministic) | ✗ |
| 5 | **LSTM v5.4** | **+1.08** | +Trump | ~53K | 32 seeds | ✓ |
| 6 | **ElasticNet** | **+1.03** | +Trump | — | 1 (deterministic) | ✗ |
| 7 | **PLS** | **+0.87** | +Onchain | — | 1 (deterministic) | ✗ |
| 8 | **PCA Regression** | **+0.56** | All | — | 1 (deterministic) | ✗ |
| 9 | **TFT v5.4** | **+0.48** | +Trump | ~95K | 32 seeds | ✓ |
| — | **VW Market** [PRIMARY] | −0.17 | — | — | — | — |
| — | **EW Market** [reference] | −0.60 | — | — | — | — |

**關鍵洞見：**

1. **模型複雜度 ≠ 排序能力**：OLS 最佳 SR=+1.31 是 LSTM v5.4 最佳 (+1.08) 的 1.2 倍、TFT v5.4 最佳 (+0.48) 的 2.7 倍；在 Price+Technical 基礎特徵組上差距更大（OLS +1.13 vs LSTM +0.72 vs TFT −1.12）
2. **非線性特徵交互有條件地有價值**：RF (+1.58) 和 GBT (+1.61) 最佳成績均來自 +Trump 特徵組，在 All features 上 RF=+1.53 微幅領先 OLS (+1.24)，但 GBT All=+1.16 反而低於 OLS，顯示非線性交互的增益依賴特徵集組合
3. **時間序列建模是負擔而非資產**：時間序列模型（LSTM, TFT）需要學習額外的時序模式，但這些模式在週頻加密市場中不穩定，反而增加了過擬合風險
4. **降維方法（PCA, PLS）不適合排序任務**：最大方差方向不一定是最佳排序方向
5. **VW Market SR=−0.174，EW Market SR=−0.596** 確認測試期市場整體下跌（VW 即 BTC/ETH 為主的大市值組合亦為負），所有模型的正 SR 均來自橫截面排序而非市場漲幅

### 7.6 核心結論：橫截面模型完勝時間序列模型

| Criterion | Gated FFN (v3) | RF | GBT | OLS | LSTM v5.4 | TFT v5.4 |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| **Best SR (PW)** | **+2.90** ✓✓ | **+1.58** | **+1.61** | **+1.31** | **+1.08** | **+0.48** |
| **Ensemble seeds** | — | **5** | **5** | 1 | **32** | **32** |
| **Params** | ~4K | — | — | — | ~53K | ~95K |
| **需要 GPU** | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ |
| **需要時序窗口** | ✗ | ✗ | ✗ | ✗ | ✓ (L=8) | ✓ (L=8) |

> 對於週頻加密貨幣橫截面報酬排序，**當期橫截面特徵足以產生穩定的排序信號**。時間序列建模不僅未能提供增量價值，反而引入了額外的過擬合風險和計算成本。最簡單的 OLS 即可超越精心設計的 LSTM+CrossAttention+ListNet 架構。

### 7.7 改善的價值：診斷性洞見

儘管 v5 的 LSTM/TFT 未能超越傳統模型，改善過程提供了有價值的診斷性洞見：

1. **Ranking Loss 有效**：將訓練目標對齊評估指標，確實改善了排序品質（LSTM Price+Tech: v4 約 −1.25 → v5.4 +0.72）
2. **Cross-Asset Attention 有效**：讓獨立處理的時間序列模型獲得跨資產比較能力，改善了排序
3. **降低容量有效**：18K 參數的 LSTM v5 優於 82K 參數的 LSTM v4，驗證了過擬合是 v4 的主要問題
4. **但根本限制仍在**：即便有排序損失和跨資產注意力，時序模型仍需通過「先獨立編碼，後跨資產比較」的兩步流程，增加了不必要的複雜度
5. **傳統模型基準不可或缺**：若未加入 OLS/RF/GBT 基準，我們可能錯誤地將 LSTM v5.4 的 SR=+0.89 視為有意義的成果

---

## 8. Limitations / 研究限制

1. **Short test period**: 49 weeks of out-of-sample evaluation
2. **Single lookback**: Only L=8 tested; {4, 12, 24} may yield different results
3. **ListNet temperature**: Only τ=1.0 tested; different temperatures may change ranking sharpness
4. **Cross-Asset Attention is simple**: Only 1 layer of self-attention; deeper cross-asset modeling (e.g., Transformer blocks) may improve ranking
5. **No hybrid approach**: Could combine ListNet on cross-sectional Gated FFN with temporal features, potentially getting best of both worlds
6. **Trump data coverage gap**: Trump 特徵僅覆蓋 64.5% 的週次（2022-03 起）。v5.4 透過 Trump-aware 訓練起點緩解了此問題，但訓練集較小（104 vs 218 batches）
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

## 10. v4 → v5.3 Changes Summary / 改版摘要

| Component | v4 | v5.4 | v5.3 | v5.4 |
|-----------|----|----|------|------|
| **Loss function** | MSE | **ListNet ranking loss** | — | — |
| **Training paradigm** | Random (asset, time) pairs | **Cross-sectional batches** | — | — |
| **Cross-asset** | None | **Self-Attention (2 heads)** | — | — |
| **VSN** | Full GRN (55K params for M=33) | **LightVSN (5K params)** | — | — |
| **Hidden dim** | 64 | 32 | — | **64**（恢復，v5.3 過於保守） |
| **LSTM layers** | 2 | **1** | — | — |
| **Dropout** | 0.15 | **0.30** | — | — |
| **Weight decay** | 1e-5 | **1e-4** | — | — |
| **Lookback** | 12 | **8** | — | — |
| **Position encoding** | None | **Learnable** | — | — |
| **LR warmup** | None | **10 epochs linear** | — | — |
| **Num seeds** | 8 | **32** | — | — |
| **Deterministic** | No | **Yes (CUDA deterministic)** | — | — |
| **Trump-aware train** | N/A | **Auto-detect start week** | — | — |
| **Traditional baselines** | N/A | N/A | OLS, EN, PCA, PLS, RF, GBT | — |
| **RF/GBT ensemble** | N/A | N/A | 1 model (best HP) | **5-seed ensemble** |
| **Market benchmark** | N/A | N/A | EW Market | **VW Market [PRIMARY]**（CoinGecko 實際市值） |
| **Look-ahead bias** | N/A | N/A | **Verified: none** | — |
| **Cross-model compare** | N/A | N/A | **10 models × 4 configs** | — |

---

## 11. File Structure / 檔案結構

```
machine_learning_for_crypto_TFT/
├── README.md                   # This file
├── config.json                 # Hyperparameters, feature configs, traditional model grids
├── data_loader.py              # Load btc_panel.npz → cross-sectional batches
├── models.py                   # LSTM + TFT with Cross-Asset Attention
├── train.py                    # ListNet ranking loss training loop (DL models)
├── train_traditional.py        # Traditional ML baselines (OLS, EN, PCA, PLS, RF, GBT)
├── evaluate.py                 # Portfolio evaluation, comparison tables, visualization
├── run_all.sh                  # One-click: train all → evaluate → compare
├── checkpoints/
│   ├── tft_results.npz         # TFT ensemble predictions
│   ├── lstm_results.npz        # LSTM ensemble predictions
│   ├── ols_results.npz         # OLS predictions
│   ├── elasticnet_results.npz  # ElasticNet predictions
│   ├── pca_regression_results.npz
│   ├── pls_results.npz
│   ├── random_forest_results.npz
│   ├── gradient_boosting_results.npz
│   └── market_portfolio.npz    # EW/VW market portfolio benchmark
└── outputs/
    ├── cross_model_comparison.csv  # Full model comparison table
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
