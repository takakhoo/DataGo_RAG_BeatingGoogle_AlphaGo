#!/bin/bash
# Launch script for Phase 2 tuning (offline JSON workflow)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TUNING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TUNING_DIR}/.." && pwd)"
cd "${TUNING_DIR}"
export PYTHONPATH="${TUNING_DIR}:${PYTHONPATH:-}"

echo "=========================================="
echo "RAG-AlphaGo Phase 2 Tuning Launcher"
echo "=========================================="
echo ""

OUTPUT_DIR="./tuning_results/phase2"
DEEP_DIR="${OUTPUT_DIR}/deep_mcts"
RECURSION_DIR="${OUTPUT_DIR}/recursion"
POSITIONS_JSON_DIR="./tuning_results/offline_positions"

PHASE1_CONFIG="./tuning_results/phase1/best_config_phase1.json"
PHASE1B_CONFIG="./tuning_results/phase1b/storage_threshold_results.json"

if [ ! -f "${PHASE1_CONFIG}" ]; then
    echo "Error: Phase 1 config not found at ${PHASE1_CONFIG}"
    echo "Run ./phase1/run_phase1.sh first."
    exit 1
fi

if [ ! -f "${PHASE1B_CONFIG}" ]; then
    echo "Error: Phase 1b storage config not found at ${PHASE1B_CONFIG}"
    exit 1
fi

if [ ! -d "${POSITIONS_JSON_DIR}" ]; then
    echo "Error: offline positions directory not found at ${POSITIONS_JSON_DIR}"
    echo "Populate it with JSON logs from Phase 1 offline analysis"
    echo "or run python phase2/make_dummy_offline_positions.py to create dummy data."
    exit 1
fi

mkdir -p "${DEEP_DIR}" "${RECURSION_DIR}"

echo "Checking Python dependencies..."
python - <<'PYTHON'
import importlib, sys
for pkg in ("numpy", "matplotlib", "psutil"):
    if importlib.util.find_spec(pkg) is None:
        sys.exit(f"Missing required package: {pkg}")
PYTHON
echo "Dependencies OK."
echo ""

echo "Starting monitoring dashboard..."
python phase2/monitor.py \
    --mode monitor \
    --results-dir ./tuning_results \
    --phase phase2 \
    --refresh 15 &
MONITOR_PID=$!
sleep 2

echo "[Phase 2a] Deep MCTS tuning (offline JSON)..."
python phase2/phase2_deep_mcts.py \
    --output-dir "${DEEP_DIR}" \
    --positions-json-dir "${POSITIONS_JSON_DIR}" \
    --phase1-config "${PHASE1_CONFIG}" \
    --storage-config "${PHASE1B_CONFIG}"
DEEP_EXIT_CODE=$?

if [ ${DEEP_EXIT_CODE} -ne 0 ]; then
    echo "Phase 2a failed (exit ${DEEP_EXIT_CODE})."
    kill "${MONITOR_PID}" 2>/dev/null || true
    exit ${DEEP_EXIT_CODE}
fi

echo "[Phase 2b] Recursion tuning (offline JSON)..."
python phase2/phase2_recursion.py \
    --output-dir "${RECURSION_DIR}" \
    --positions-json-dir "${POSITIONS_JSON_DIR}" \
    --phase1-config "${PHASE1_CONFIG}" \
    --storage-config "${PHASE1B_CONFIG}" \
    --deep-mcts-results "${DEEP_DIR}/phase2_deep_mcts_results.json"
RECURSION_EXIT_CODE=$?

echo "Stopping monitor..."
kill "${MONITOR_PID}" 2>/dev/null || true

if [ ${RECURSION_EXIT_CODE} -ne 0 ]; then
    echo "Phase 2b failed (exit ${RECURSION_EXIT_CODE})."
    exit ${RECURSION_EXIT_CODE}
fi

echo ""
echo "=========================================="
echo "Phase 2 offline tuning completed."
echo "=========================================="
echo "Deep MCTS summary : ${DEEP_DIR}/phase2_deep_mcts_results.json"
echo "Recursion summary : ${RECURSION_DIR}/phase2_recursion_results.json"
echo ""

