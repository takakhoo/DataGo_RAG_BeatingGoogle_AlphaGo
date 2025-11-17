#!/bin/bash
# Quick competitive test: 3 games to validate performance advantage
set -e

echo "======================================================================"
echo "DataGo vs KataGo - Quick Test (3 games)"
echo "======================================================================"
echo ""

cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"

# Activate virtual environment
source /scratch2/f004ndc/AlphaGo\ Project/Go_env/bin/activate

# Run 3 quick games
python3 run_datago_recursive_match.py \
    --katago-executable "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago" \
    --katago-model "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz" \
    --katago-config "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg" \
    --config "src/bot/config.yaml" \
    --games 3 \
    --max-moves 100 \
    2>&1 | tee quick_test_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "Quick test complete!"
