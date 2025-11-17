#!/bin/bash
# Run competitive match: DataGo vs KataGo
# DataGo uses 800 visits standard + 2000 visits deep search
# KataGo uses 800 visits constant

set -e

echo "======================================================================"
echo "DataGo vs KataGo Competitive Match"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  DataGo:  800 visits (standard) + 2000 visits (deep search on 5% moves)"
echo "  KataGo:  800 visits (constant)"
echo "  Games:   10"
echo "  Max moves per game: 200"
echo ""
echo "Expected: DataGo should win more due to adaptive deep search"
echo "======================================================================"
echo ""

cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"

# Activate virtual environment
source /scratch2/f004ndc/AlphaGo\ Project/Go_env/bin/activate

# Run the match
python3 run_datago_recursive_match.py \
    --katago-executable "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago" \
    --katago-model "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz" \
    --katago-config "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg" \
    --config "src/bot/config.yaml" \
    --games 10 \
    --max-moves 200 \
    2>&1 | tee competitive_match_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "======================================================================"
echo "Match complete! Check the log file for detailed results."
echo "======================================================================"
