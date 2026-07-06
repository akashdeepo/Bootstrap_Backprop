"""
Configuration for the amortized bootstrap project (v2 redesign).

Design principles (see RESEARCH_PLAN.md):
  - F is random: every training example draws fresh parameters from a prior,
    so the target distribution varies with the input (no constant shortcut).
  - Leakage-proof seed layout: train / val / test / torch streams are
    spawned independently. Test-time ground truth is computed analytically
    or from draws that never touch training.
"""

from pathlib import Path
import numpy as np
from numpy.random import SeedSequence, default_rng

# ----------------------------------------------
# Paths
# ----------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------
# Global parameters
# ----------------------------------------------
N = 200                  # observations per dataset
SEED = 42

# Quantile levels predicted by the model: 0.005, 0.010, ..., 0.995.
# Includes 0.025/0.975 and 0.05/0.95 needed for 95% and 90% intervals.
QUANTILE_LEVELS = np.round(np.arange(0.005, 0.9951, 0.005), 4)
N_LEVELS = len(QUANTILE_LEVELS)  # 199

# ----------------------------------------------
# Seed management
# ----------------------------------------------
# streams[0] -> training example generation
# streams[1] -> validation example generation
# streams[2] -> test parameter + dataset generation
# streams[3] -> torch initialization / shuffling seed source
# streams[4] -> diagnostics (noise inputs etc.)
# streams[5] -> ground-truth quantile tables (stable MC table)
# streams[6] -> baseline resampling (bootstrap index draws)
# NOTE: appending streams is safe -- SeedSequence children are deterministic
# by index, so streams 0-4 are bit-identical to earlier runs. Never reorder.
_ss = SeedSequence(SEED)
_children = _ss.spawn(7)
RNG_TRAIN = default_rng(_children[0])
RNG_VAL = default_rng(_children[1])
RNG_TEST = default_rng(_children[2])
TORCH_SEED = int(_children[3].generate_state(1)[0])
RNG_DIAG = default_rng(_children[4])
RNG_TABLE = default_rng(_children[5])
RNG_BASELINE = default_rng(_children[6])

# Directory for large regenerable caches (gitignored)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
