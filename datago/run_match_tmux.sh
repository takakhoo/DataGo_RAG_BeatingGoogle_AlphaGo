#!/bin/bash
# Run DataGo vs KataGo match in tmux

SESSION_NAME="datago_match_gpu7"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME

# Send commands to the session
tmux send-keys -t $SESSION_NAME "cd '/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo'" C-m
tmux send-keys -t $SESSION_NAME "source ../Go_env/bin/activate" C-m
tmux send-keys -t $SESSION_NAME "python run_datago_match.py \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-executable ../KataGo/cpp/katago \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-model ../KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz \\" C-m
tmux send-keys -t $SESSION_NAME "  --katago-config ../KataGo/configs/gtp_800visits.cfg \\" C-m
tmux send-keys -t $SESSION_NAME "  --config src/bot/config.yaml \\" C-m
tmux send-keys -t $SESSION_NAME "  --games 3 \\" C-m
tmux send-keys -t $SESSION_NAME "  --max-moves 250" C-m

echo "Tmux session '$SESSION_NAME' created!"
echo "Attach with: tmux attach -t $SESSION_NAME"
echo "View output with: tmux capture-pane -t $SESSION_NAME -p"
