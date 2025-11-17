#!/bin/bash
# Test script for recursive deep search

cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"

# Activate environment
source "../Go_env/bin/activate"

# Run a short test game
python run_datago_recursive_match.py \
    --katago-executable "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago" \
    --katago-model "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz" \
    --katago-config "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg" \
    --config "src/bot/config.yaml" \
    --games 1 \
    --max-moves 40
