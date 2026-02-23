"""
Genie (Offline Optimal) Baseline — Backward Dynamic Programming.

Uses the EXACT same QoE formula as evaluation:
    QoE_k = VMAF(a_k) - 4.3 * rebuf_k - 1.0 * |VMAF(a_k) - VMAF(a_{k-1})|

State space: (chunk_index, last_action, discretized_buffer)
With 48 chunks × 6 actions × 121 buffer levels ≈ 35k states → solves in < 1 second.
"""

import numpy as np


class Genie:
    def __init__(self, env):
        self.env = env
        self.bitrate_levels = env.BITRATE_LEVELS
        self.n_actions = len(self.bitrate_levels)
        self.chunk_duration = env.CHUNK_DURATION
        self.buffer_max = env.BUFFER_MAX
        self.max_chunks = env.max_chunks
        self.rebuf_penalty = env.REBUF_PENALTY_BASE      # 4.3
        self.smooth_penalty = env.SMOOTH_PENALTY_WEIGHT   # 1.0
        self.min_tp = env.MIN_NETWORK_THROUGHPUT
        self.max_tp = env.MAX_NETWORK_THROUGHPUT

        self.buf_step = 0.25
        self.n_buf = int(self.buffer_max / self.buf_step) + 1

        self._policy = None
        self._last_action = 0

    def _buf_idx(self, buf):
        return int(np.clip(round(buf / self.buf_step), 0, self.n_buf - 1))

    def select_bitrate(self, chunk_idx, buffer_level, trace_throughput):
        if chunk_idx == 0:
            self._solve_dp(trace_throughput)
            self._last_action = 0

        bi = self._buf_idx(buffer_level)
        action = int(self._policy[chunk_idx, self._last_action, bi])
        self._last_action = action
        return action

    def _solve_dp(self, trace_tp):
        n_a = self.n_actions
        n_k = self.max_chunks
        vmaf_map = self.env.vmaf_scores

        vmaf_vals = np.array([vmaf_map.get(int(br), 35.0) for br in self.bitrate_levels])

        dl_times = np.zeros((n_k, n_a))
        for k in range(n_k):
            tp_idx = int(k * self.chunk_duration) % len(trace_tp)
            tp = np.clip(trace_tp[tp_idx], self.min_tp, self.max_tp)
            for a in range(n_a):
                chunk_bits = self.bitrate_levels[a] * 1000 * self.chunk_duration
                dl = chunk_bits / (tp * 1000.0)
                dl_times[k, a] = min(dl, 60.0)

        V = np.zeros((n_a, self.n_buf))
        self._policy = np.zeros((n_k, n_a, self.n_buf), dtype=np.int32)

        for k in range(n_k - 1, -1, -1):
            V_new = np.full((n_a, self.n_buf), -1e9)

            for last_a in range(n_a):
                for bi in range(self.n_buf):
                    buf = bi * self.buf_step
                    best_val = -1e9
                    best_act = 0

                    for a in range(n_a):
                        dl = dl_times[k, a]
                        rebuf = max(0.0, dl - buf)
                        new_buf = min(max(0.0, buf - dl) + self.chunk_duration, self.buffer_max)
                        new_bi = self._buf_idx(new_buf)

                        reward = (
                            vmaf_vals[a]
                            - self.rebuf_penalty * rebuf
                            - self.smooth_penalty * abs(vmaf_vals[a] - vmaf_vals[last_a])
                        )

                        val = reward + V[a, new_bi]
                        if val > best_val:
                            best_val = val
                            best_act = a

                    V_new[last_a, bi] = best_val
                    self._policy[k, last_a, bi] = best_act

            V = V_new
