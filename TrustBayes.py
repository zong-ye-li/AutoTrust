"""Count-based posterior utility; mathematical queries are interior-only."""
import re
import numpy as np
import pandas as pd

class TrustBayesCalculator:
    def __init__(self, dist_csv_path, deltaV_csv_path=None, alpha=9., smoothing=0., rng=None):
        if not np.isfinite(alpha) or alpha < 0 or not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError('Invalid alpha/smoothing')
        self.alpha, self.smoothing = float(alpha), float(smoothing)
        self.rng = np.random.default_rng() if rng is None else rng
        self.scores = np.arange(1, 11)
        df = pd.read_csv(dist_csv_path)
        bins = []
        for value in df.Score_Interval:
            nums = re.findall(r'-?\d+(?:\.\d+)?', str(value))
            if len(nums) != 2:
                raise ValueError('Expected interval score bins')
            lo, hi = map(float, nums)
            if hi - lo != 1:
                raise ValueError('Invalid bin width')
            bins.append((lo + hi) / 2)
        order = np.argsort(bins)
        if not np.array_equal(np.array(bins)[order], self.scores):
            raise ValueError('Expected exactly ten unique bins 1..10')
        for column, attr in [('Count_Benign', 'p_s_given_B'), ('Count_Malicious', 'p_s_given_M')]:
            counts = df[column].to_numpy(dtype=float)[order]
            if not np.isfinite(counts).all() or (counts < 0).any() or (counts != np.floor(counts)).any() or counts.sum() <= 0:
                raise ValueError('Invalid class counts')
            counts = counts + smoothing
            setattr(self, attr, counts / counts.sum())
        self.deltaV_table = None
        if deltaV_csv_path is not None and smoothing == 0 and alpha == 9:
            table = pd.read_csv(deltaV_csv_path)
            interior = table[(table.Prior_c_B > 0) & (table.Prior_c_B < 1)]
            expected = [self.moments(c)['delta_v'] for c in interior.Prior_c_B]
            if len(interior) == 0 or not np.allclose(interior.Delta_V, expected, atol=5.1e-7, rtol=0):
                raise ValueError('Unverified Delta-V cache')
            self.deltaV_table = interior

    def _utilities(self, c):
        if not np.isfinite(c) or not 0 < c < 1:
            raise ValueError('Credence must be finite and strictly interior (0,1)')
        marginal = c * self.p_s_given_B + (1-c) * self.p_s_given_M
        support = marginal > 0
        result = np.zeros(10)
        result[support] = (1+self.alpha)*c*self.p_s_given_B[support]/marginal[support]-self.alpha
        return result

    def moments(self, c):
        v = self._utilities(c)
        b, m = float(self.p_s_given_B @ v), float(self.p_s_given_M @ v)
        delta = b-m
        valid = b > 0 > m and delta > 0
        return dict(mu_b=b, mu_m=m, delta_v=delta, r0=b/delta if valid else None,
                    r0_valid=valid)

    def get_marginal_attack_cost(self, prior_c_B):
        return self.moments(prior_c_B)['delta_v']

    def get_v_score(self, prior_c_B, is_malicious_action):
        utilities = self._utilities(prior_c_B)
        pmf = self.p_s_given_M if is_malicious_action else self.p_s_given_B
        forced = getattr(self, '_forced_uniform', None)
        if forced is None:
            u = self.rng.random()
        else:
            u = float(forced)
            self._forced_uniform = None
            if not 0 <= u < 1:
                raise ValueError('Forced RNG uniform must be in [0,1)')
        idx = min(int(np.searchsorted(np.cumsum(pmf), u, side='right')), 9)
        return idx+1, float(utilities[idx])
