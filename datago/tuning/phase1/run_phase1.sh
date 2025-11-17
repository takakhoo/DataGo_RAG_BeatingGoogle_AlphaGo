#!/bin/bash
# Launch script for Phase 1 tuning with monitoring
# Usage: ./run_phase1.sh

echo "=========================================="
echo "RAG-AlphaGo Phase 1 Tuning Launcher"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="./tuning_results/phase1"
NUM_GAMES=150
PARALLEL_WORKERS=32
EARLY_STOPPING_GAMES=100
EARLY_STOPPING_THRESHOLD=0.40

# Check if output directory exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "Warning: Output directory $OUTPUT_DIR already exists."
    read -p "Continue and overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,temperature.gpu --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. Cannot verify GPU."
    echo ""
fi

# Check Python dependencies
echo "Checking dependencies..."
python -c "import numpy, matplotlib, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Error: Required Python packages not found."
    echo "Install with: pip install -r requirements_tuning.txt"
    exit 1
fi

echo "Dependencies OK"
echo ""

# Start monitoring in background
echo "Starting monitoring dashboard..."
python monitor.py \
    --mode monitor \
    --results-dir ./tuning_results \
    --phase phase1 \
    --refresh 10 &

MONITOR_PID=$!
echo "Monitor started (PID: $MONITOR_PID)"
echo ""

# Give monitor time to start
sleep 2

# Start tuning
echo "Starting Phase 1a tuning..."
echo "Configuration:"
echo "  Output dir: $OUTPUT_DIR"
echo "  Games per config: $NUM_GAMES"
echo "  Parallel workers: $PARALLEL_WORKERS"
echo "  Early stopping: $EARLY_STOPPING_GAMES games @ ${EARLY_STOPPING_THRESHOLD} win rate"
echo ""
echo "This will take approximately 18-20 hours."
echo ""

python phase1_uncertainty_tuning.py \
    --output-dir "$OUTPUT_DIR" \
    --num-games $NUM_GAMES \
    --parallel-workers $PARALLEL_WORKERS \
    --early-stopping-games $EARLY_STOPPING_GAMES \
    --early-stopping-threshold $EARLY_STOPPING_THRESHOLD

TUNING_EXIT_CODE=$?

# Stop monitoring
echo ""
echo "Stopping monitor..."
kill $MONITOR_PID 2>/dev/null

if [ $TUNING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Phase 1a completed successfully!"
    echo "=========================================="
    echo ""
    echo "Best config saved to: $OUTPUT_DIR/best_config_phase1.json"
    echo ""
    echo "Next steps:"
    echo "1. Review results: python monitor.py --mode summary --results-file $OUTPUT_DIR/phase1_results.json"
    echo "2. Run Phase 1b: ./run_phase1b.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Error: Phase 1a failed with exit code $TUNING_EXIT_CODE"
    echo "=========================================="
    echo ""
    exit $TUNING_EXIT_CODE
fi
