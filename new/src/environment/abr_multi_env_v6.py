"""
Multi-Video ABR Environment V6 (Constrained MDP)

V6 is a light-weight refinement of V5 that only adjusts a few
reward-related hyperparameters to better align training with the
evaluation QoE metric, while keeping the state representation and
transition dynamics identical.

Changes over V5:
1. Rebuffer penalty base: 6.0 → 5.0
   - Closer to the evaluation-time rebuffer weight (4.3), reducing
     train/eval mismatch and avoiding over-conservative policies.
2. Buffer deviation weight: 0.05 → 0.03
   - Slightly relaxes the pressure to hover exactly around the target
     buffer, encouraging more bitrate ambition when conditions allow.
3. Lyapunov parameters: B_REF=10 → 8, LYAPUNOV_BETA=1.0 → 0.8
   - Weakens the Lyapunov penalty so that the stability constraint
     does not dominate the primary QoE objective.

All other behavior (observation space, VBR model, trace handling,
reward decomposition in info dict, etc.) is inherited from V5.
"""

from .abr_multi_env_v5 import ABREnv as ABREnvV5


class ABREnv(ABREnvV5):
    """
    V6 variant of the multi-video ABR environment.

    Only the following class-level hyperparameters are changed relative
    to `abr_multi_env_v5.ABREnv`:

    - REBUF_PENALTY_BASE
    - BUFFER_DEV_WEIGHT
    - B_REF
    - LYAPUNOV_BETA
    """

    REBUF_PENALTY_BASE = 5.0
    BUFFER_DEV_WEIGHT = 0.03

    B_REF = 8.0
    LYAPUNOV_BETA = 0.8

