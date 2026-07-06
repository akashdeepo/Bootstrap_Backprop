"""
Configuration for the amortized bootstrap project (v2 redesign).

Design principles (see RESEARCH_PLAN.md):
  - F is random: every training example draws fresh parameters from a prior,
    so the target distribution varies with the input (no constant shortcut).
  - Leakage-proof seed layout: train / val / test / torch streams are
    spawned independently. Test-time ground truth is computed analytically
    or from draws that never touch training.
"""

import os
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
# streams[7] -> M4 universal-model training data
# streams[8] -> M4 universal-model validation data
# streams[9] -> M4 out-of-family test data
# streams[10] -> M4 truth pools / spare
# NOTE: appending streams is safe -- SeedSequence children are deterministic
# by index, so streams 0-6 are bit-identical to earlier runs. Never reorder.
N_STREAMS = 12
_ss = SeedSequence(SEED)
_children = _ss.spawn(N_STREAMS)
RNG_TRAIN = default_rng(_children[0])
RNG_VAL = default_rng(_children[1])
RNG_TEST = default_rng(_children[2])
TORCH_SEED = int(_children[3].generate_state(1)[0])
RNG_DIAG = default_rng(_children[4])
RNG_TABLE = default_rng(_children[5])
RNG_BASELINE = default_rng(_children[6])
RNG_M4_TRAIN = default_rng(_children[7])
RNG_M4_VAL = default_rng(_children[8])
RNG_M4_OOD = default_rng(_children[9])
RNG_M4_POOL = default_rng(_children[10])


def fresh_rng(stream_index: int):
    """A NEW generator at the start of the given stream. Lets a later
    experiment reproduce another script's draws exactly (e.g. the
    universal model regenerating each specialist's test set)."""
    return default_rng(SeedSequence(SEED).spawn(N_STREAMS)[stream_index])


# ----------------------------------------------
# Multi-seed replication (AB_VARIANT environment variable)
# ----------------------------------------------
# Variant runs re-randomize ONLY the training data, validation data, and
# torch initialization. Test sets, ground-truth caches, and baseline
# resampling stay on the base streams, so replications are PAIRED
# comparisons on identical test data and reuse all expensive caches.
# Experiments append VTAG to their output filenames.
VARIANT = int(os.environ.get('AB_VARIANT', '0'))
VTAG = '' if VARIANT == 0 else f'_v{VARIANT}'
if VARIANT:
    _vchildren = SeedSequence((SEED, VARIANT)).spawn(4)
    RNG_TRAIN = default_rng(_vchildren[0])
    RNG_VAL = default_rng(_vchildren[1])
    TORCH_SEED = int(_vchildren[2].generate_state(1)[0])
    RNG_DIAG = default_rng(_vchildren[3])

# Directory for large regenerable caches (gitignored)
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
