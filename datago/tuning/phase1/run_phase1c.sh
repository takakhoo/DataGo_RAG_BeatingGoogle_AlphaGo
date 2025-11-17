#!/bin/bash
# Launch script for Phase 1c uncertainty threshold tuning
# Usage: ./run_phase1c.sh

echo "=========================================="
echo "RAG-AlphaGo Phase 1c: Uncertainty Threshold Tuning"
echo "=========================================="
echo ""

# Configuration
PHASE1A_CONFIG="${1:-./tuning_results/phase1a/best_uncertainty_config.json}"
PHASE1B_WEIGHTS="${2:-./tuning_results/phase1b/phase1b_results.json}"
RAG_DATABASE="${3:-./data/rag_database.json}"
GROUND_TRUTH_DB="${4:-./data/shallow_mcts_db.json}"
OUTPUT_DIR="${5:-./tuning_results/phase1c}"
NUM_GAMES="${6:-100}"

# Check if Phase 1a config exists
if [ ! -f "$PHASE1A_CONFIG" ]; then
    echo "Error: Phase 1a config not found at $PHASE1A_CONFIG"
    echo "Please run Phase 1a first to get uncertainty function parameters"
    exit 1
fi

# Check if Phase 1b weights exist
if [ ! -f "$PHASE1B_WEIGHTS" ]; then
    echo "Error: Phase 1b weights not found at $PHASE1B_WEIGHTS"
    echo "Please run Phase 1b first to get relevance weights"
    exit 1
fi

# Check if RAG database exists
if [ ! -f "$RAG_DATABASE" ]; then
    echo "Error: RAG database not found at $RAG_DATABASE"
    echo "Please populate RAG database with deep MCTS results"
    exit 1
fi

# Check if ground truth database exists
if [ ! -f "$GROUND_TRUTH_DB" ]; then
    echo "Error: Ground truth database not found at $GROUND_TRUTH_DB"
    echo "Need this for percentile estimation"
    exit 1
fi

echo "Using Phase 1a config: $PHASE1A_CONFIG"
echo "Using Phase 1b weights: $PHASE1B_WEIGHTS"
echo "Using RAG database: $RAG_DATABASE"
echo "Using ground truth DB: $GROUND_TRUTH_DB"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start Phase 1c tuning
echo "Starting Phase 1c tuning..."
echo "Configuration:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Games per threshold: $NUM_GAMES"
echo ""
echo "This will take approximately 8-10 hours."
echo ""

python tuning/phase1/phase1c_uncertainty_threshold.py \
    --phase1a-config "$PHASE1A_CONFIG" \
    --phase1b-weights "$PHASE1B_WEIGHTS" \
    --rag-database "$RAG_DATABASE" \
    --ground-truth-db "$GROUND_TRUTH_DB" \
    --output-dir "$OUTPUT_DIR" \
    --num-games "$NUM_GAMES"

TUNING_EXIT_CODE=$?

# Check results
if [ $TUNING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 1c tuning completed successfully!"
    echo "=========================================="
    echo ""
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Next steps:"
    echo "  1. Review results in $OUTPUT_DIR/phase1c_results.json"
    echo "  2. Check plots in $OUTPUT_DIR/"
    echo "  3. Use best threshold for production gameplay"
else
    echo ""
    echo "=========================================="
    echo "Phase 1c tuning failed with exit code $TUNING_EXIT_CODE"
    echo "=========================================="
    exit $TUNING_EXIT_CODE
fi
fi
