#!/bin/bash
# Run Phase 1b: Relevance Weight Tuning
# Tests how policy/value change when features are perturbed for identical game states

set -e

# Configuration
RAG_DATABASE="${1:-./data/rag_database.json}"
OUTPUT_DIR="${2:-./tuning_results/phase1b}"
METHOD="${3:-grid}"  # grid or optimize
MIN_GROUP_SIZE="${4:-3}"

echo "=================================="
echo "Phase 1b: Relevance Weight Tuning"
echo "=================================="
echo "RAG Database: $RAG_DATABASE"
echo "Output: $OUTPUT_DIR"
echo "Method: $METHOD"
echo "Min group size: $MIN_GROUP_SIZE"
echo "=================================="

# Check if database exists
if [ ! -f "$RAG_DATABASE" ]; then
    echo "Error: RAG database not found at $RAG_DATABASE"
    echo ""
    echo "You need a RAG database with positions that have:"
    echo "  - Multiple entries with IDENTICAL sym_hash (same game state)"
    echo "  - Different contexts (komi, winrate, etc.)"
    echo ""
    echo "This allows testing how non-gamestate features affect policy/value."
    exit 1
fi

# Run Phase 1b tuning
python tuning/phase1/phase1b_relevance_weights.py \
    --rag-database "$RAG_DATABASE" \
    --output-dir "$OUTPUT_DIR" \
    --method "$METHOD" \
    --min-group-size "$MIN_GROUP_SIZE"

echo ""
echo "✓ Phase 1b complete!"
echo "Results saved to: $OUTPUT_DIR/phase1b_results.json"
echo ""
echo "Next step: Use these weights in Phase 1c"
