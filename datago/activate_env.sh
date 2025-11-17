#!/bin/bash
# Quick activation script for Go_env
# Usage: source activate_env.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
GO_ENV_PATH="$SCRIPT_DIR/../Go_env"

if [ -f "$GO_ENV_PATH/bin/activate" ]; then
    source "$GO_ENV_PATH/bin/activate"
    echo "✓ Go_env activated"
    echo "Python: $(which python)"
    echo "Location: $GO_ENV_PATH"
else
    echo "Error: Go_env not found at $GO_ENV_PATH"
    exit 1
fi
