#!/bin/bash
# test_katago_gpu7.sh
# Quick test to verify KataGo works on GPU 7

echo "Testing KataGo on GPU 7..."
echo "================================"

cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"

# Test with the updated config
"/scratch2/f004ndc/AlphaGo Project/Go_env/bin/python" test_gtp_katago.py 2>&1 | tail -30
