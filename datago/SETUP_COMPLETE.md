# Go Bot vs KataGo Setup - Complete

## Summary

Successfully set up your Go bot to play against KataGo without the gomill dependency (which is Python 2 only). Here's what was completed:

## What Was Done

### 1. ✅ Compiled KataGo
- **Location**: `/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago`
- **Backend**: OpenCL (for GPU acceleration)
- **Version**: KataGo v1.16.4
- **Status**: Compiled and GPU-tuned successfully

### 2. ✅ Downloaded Neural Network Model
- **Location**: `/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz`
- **Source**: KataGo test models (10-block network)
- **Note**: You can download stronger models from https://katagotraining.org/ if needed

### 3. ✅ Created KataGo Configuration
- **Location**: `/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg`
- **Settings**: Configured for 800 visits per move (matching your bot's settings)

### 4. ✅ Removed gomill Dependency
Replaced gomill (Python 2 only) with custom GTP implementation:

#### New Files Created:
- **`src/bot/gtp_controller.py`**: Custom GTP controller for communicating with Go engines
- **`src/bot/gtp_player.py`**: Wrapper for DataGo bot to play via GTP
- **`test_gtp_katago.py`**: Standalone test script to verify KataGo GTP interface

#### Files Modified:
- **`play_vs_katago.py`**: Rewritten to use custom GTP instead of gomill
- **`src/bot/__init__.py`**: Updated imports to use GTPPlayer instead of GomillPlayer
- **`src/bot/config.yaml`**: Updated with correct KataGo paths

#### Files You Can Delete:
- **`src/bot/gomill_player.py`**: No longer needed (replaced by gtp_player.py)

### 5. ✅ Tested KataGo GTP Interface
- Test script confirms KataGo responds to GTP commands correctly
- Can generate moves and play games
- GPU tuning completed successfully

## How to Use

### Test KataGo GTP Interface
```bash
cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"
python test_gtp_katago.py
```

### Play DataGo vs KataGo
Once your DataGo bot implementation is complete:
```bash
cd "/scratch2/f004ndc/AlphaGo Project/RAG-MCTS-AlphaGo"
source "../Go_env/bin/activate"

python play_vs_katago.py \
    --config src/bot/config.yaml \
    --katago-executable "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago" \
    --katago-model "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz" \
    --katago-config "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg" \
    --datago-color black \
    --games 1
```

## Configuration Details

### KataGo Settings (gtp_800visits.cfg)
- **Max Visits**: 800 (same as DataGo bot)
- **Pondering**: Disabled (for fair comparison)
- **Search Threads**: 6 (adjust based on your system)
- **Rules**: Tromp-Taylor

### DataGo Bot Config (src/bot/config.yaml)
```yaml
katago:
  executable_path: "/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago"
  model_path: "/scratch2/f004ndc/AlphaGo Project/KataGo/models/g170e-b10c128-s1141046784-d204142634.bin.gz"
  config_path: "/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg"
  visits: 800
```

## Custom GTP Implementation

The custom GTP implementation is pure Python 3 and handles:
- Starting/stopping GTP engines
- Sending commands and parsing responses
- Playing moves and managing game state
- Board setup (size, komi, clear board)
- Move generation

### Key Classes:
- **`GTPController`**: Manages subprocess communication with GTP engines
- **`GTPPlayer`**: Wrapper for your DataGo bot to play via GTP protocol

## Notes

- **First Run**: KataGo will auto-tune for your GPU (~2 minutes). This happens once.
- **Model Strength**: The current model is small (10 blocks). Download larger models from katagotraining.org for stronger play.
- **Python Environment**: Using `Go_env` in AlphaGo Project directory with numpy, pyyaml, and other dependencies installed.
- **No More gomill**: All gomill dependencies removed. The codebase is now Python 3 compatible.

## Next Steps

To play games, you'll need to complete the DataGo bot implementation:
1. Ensure `DataGoBot.generate_move()` works correctly
2. Verify RAG database integration
3. Test the full pipeline with `play_vs_katago.py`

## Troubleshooting

### KataGo Takes Long to Start
- First run performs GPU tuning (~2 minutes)
- Subsequent runs are fast
- Tuning results saved in `~/.katago/`

### Import Errors
- Make sure you're in the project directory
- Activate the virtual environment: `source "../Go_env/bin/activate"`
- Or use the Python directly: `"/scratch2/f004ndc/AlphaGo Project/Go_env/bin/python"`

### Path Issues
- All paths in `config.yaml` are absolute
- Verify files exist at the specified locations
- Check permissions on KataGo executable

## Files Summary

### Created:
- `/scratch2/f004ndc/AlphaGo Project/KataGo/cpp/katago` (compiled executable)
- `/scratch2/f004ndc/AlphaGo Project/KataGo/configs/gtp_800visits.cfg`
- `src/bot/gtp_controller.py`
- `src/bot/gtp_player.py`
- `test_gtp_katago.py`

### Modified:
- `play_vs_katago.py`
- `src/bot/__init__.py`
- `src/bot/config.yaml`

### Can Delete:
- `src/bot/gomill_player.py` (replaced by gtp_player.py)
- `simple_test_katago.py` (if exists - superseded by test_gtp_katago.py)
