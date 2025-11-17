#!/bin/bash
# run_bot_vs_katago.sh
# Run DataGo bot vs KataGo on GPU 7 in tmux

SESSION_NAME="datago_vs_katago"
PROJECT_DIR="/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"
GO_ENV="/scratch2/f004ndc/AlphaGo Project/Go_env/bin/activate"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session
tmux new-session -d -s $SESSION_NAME -c "$PROJECT_DIR"

# Set up the environment and run the bot
tmux send-keys -t $SESSION_NAME "source $GO_ENV" C-m
tmux send-keys -t $SESSION_NAME "echo '=== DataGo Bot vs KataGo on GPU 7 ==='" C-m
tmux send-keys -t $SESSION_NAME "echo 'Starting match...'" C-m
tmux send-keys -t $SESSION_NAME "" C-m

# Run the play script (commented out until DataGo bot is ready)
# Uncomment this line when ready to run:
tmux send-keys -t $SESSION_NAME "python play_vs_katago.py \
  --config src/bot/config.yaml \
  --katago-executable '/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago' \
  --katago-model '/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz' \
  --katago-config '/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg' \
  --datago-color black \
  --games 1" C-m

echo "=========================================="
echo "Tmux session '$SESSION_NAME' created!"
echo "=========================================="
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach from the session:"
echo "  Press Ctrl+B, then D"
echo ""
echo "To kill the session:"
echo "  tmux kill-session -t $SESSION_NAME"
echo ""
echo "To view logs:"
echo "  tail -f logs/datago_bot.log"
echo "=========================================="
