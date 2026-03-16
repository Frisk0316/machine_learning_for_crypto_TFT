#!/usr/bin/env bash
# run_all.sh — One-click: Train TFT + LSTM → Evaluate → Compare
set -e
cd "$(dirname "$0")"

echo "═══════════════════════════════════════════════════════════"
echo "  Step 1/4: Train TFT (8 seeds × 3 info sets)"
echo "═══════════════════════════════════════════════════════════"
python train.py --config config.json --model tft

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 2/4: Train LSTM baseline (8 seeds × 3 info sets)"
echo "═══════════════════════════════════════════════════════════"
python train.py --config config.json --model lstm

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 3/4: Evaluate TFT"
echo "═══════════════════════════════════════════════════════════"
python evaluate.py --config config.json --model tft

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Step 4/4: Evaluate LSTM"
echo "═══════════════════════════════════════════════════════════"
python evaluate.py --config config.json --model lstm

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All done! Results in ./outputs/"
echo "═══════════════════════════════════════════════════════════"
