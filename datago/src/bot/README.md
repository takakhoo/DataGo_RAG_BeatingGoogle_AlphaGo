# DataGo Bot

RAG-enhanced Go bot that integrates KataGo's MCTS with retrieval-augmented generation.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r ../../requirements.txt
pip install gomill pyyaml

# Install KataGo (download from GitHub releases)
# https://github.com/lightvector/KataGo/releases
```

### Configuration

1. Edit `config.yaml` and set paths:
   - `katago.executable_path`: Path to KataGo binary
   - `katago.model_path`: Path to KataGo model
   - `katago.config_path`: Path to KataGo GTP config
   - `rag_database.database_path`: Path to RAG database

2. Use default parameters or load tuned parameters from `tuning/tuning_results/`

### Usage

#### Play Against KataGo

```bash
python gomill_player.py \
    --config config.yaml \
    --katago-executable /path/to/katago \
    --katago-model /path/to/model.bin.gz \
    --katago-config /path/to/gtp_config.cfg \
    --color black
```

#### Interactive Testing

```python
from datago_bot import DataGoBot

bot = DataGoBot("config.yaml")
bot.new_game()

for _ in range(10):
    decision = bot.generate_move()
    print(f"{decision.move} (unc={decision.uncertainty:.3f})")
    bot.game_state.move_number += 1

print(bot.get_statistics())
bot.shutdown()
```

## Files

- `datago_bot.py`: Main bot implementation
- `gomill_player.py`: Gomill integration for playing games
- `config.yaml`: Configuration file with all parameters
- `__init__.py`: Package initialization

## Documentation

See `../../BOT_IMPLEMENTATION.md` for comprehensive documentation including:
- Architecture overview
- Component details
- Configuration reference
- Implementation details
- Usage examples
- Parameter tuning guide

## Key Features

✅ **Uncertainty Detection**: Identifies complex positions using policy entropy and value variance

✅ **RAG Integration**: Queries database of deeply analyzed positions

✅ **Knowledge Blending**: Blends retrieved analysis with network priors

✅ **Deep MCTS**: Performs 10k+ visit searches on novel complex positions

✅ **Online Learning**: Stores new positions during gameplay

✅ **Gomill Integration**: Plays through gomill library

✅ **Comprehensive Config**: All parameters in config.yaml for easy tuning

## Next Steps

1. **Fix Integration**: Resolve import paths with existing codebase
2. **Test Components**: Test KataGo communication, RAG queries, etc.
3. **Parameter Tuning**: Run Phase 1-3 tuning scripts
4. **Evaluate**: Play games and measure performance

## Parameters to Tune

| Phase | Parameters | Script |
|-------|-----------|--------|
| 1a | w1, w2, phase_function | `tuning/phase1/phase1_uncertainty_tuning.py` |
| 1b | relevance_weights | `tuning/phase1/phase1b_relevance_weights.py` |
| 1c | uncertainty_threshold | `tuning/phase1/phase1c_uncertainty_threshold.py` |
| 2 | deep_mcts params | `tuning/phase2/phase2_deep_mcts.py` |
| 3 | beta, recursion_depth | Manual testing |

See `tuning/parameter_tuning_plan.txt` for full tuning strategy.
