# Differential Privacy mechanism for Federated Learning.
# Implements the Gaussian Mechanism with Rényi DP (RDP) epsilon accounting.
#
# Three-step protection:
#   1. Gradient Clipping  — bounds L2 sensitivity of each client update
#   2. Gaussian Noise     — adds calibrated noise to privatize parameters
#   3. RDP Budget Tracking — tracks cumulative (epsilon, delta)-DP guarantee
from __future__ import annotations
import numpy as np


class DPMechanism:
    """
    (ε, δ)-Differential Privacy via the subsampled Gaussian Mechanism.

    Parameters
    ----------
    clip_norm     : float  L2 norm bound for joint parameter clipping
    sigma         : float  Gaussian noise multiplier (noise_std = clip_norm * sigma)
    delta         : float  DP failure probability target
    sampling_rate : float  Poisson subsampling rate q (fraction of population
                           sampled per round, e.g. 3/100 = 0.03 for 3 clients
                           drawn from a pool of ~100).  Enables amplification
                           by subsampling: ε_RDP_sub ≈ q² × ε_RDP_base.
    """

    def __init__(
        self,
        clip_norm:     float = 1.0,
        sigma:         float = 1.0,
        delta:         float = 1e-5,
        sampling_rate: float = 0.03,
    ):
        self.clip_norm     = clip_norm
        self.sigma         = sigma
        self.delta         = delta
        self.sampling_rate = sampling_rate

    # ── Step 1: Gradient Clipping ─────────────────────────────────────────────

    def clip_parameters(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project the full parameter vector [coef ‖ intercept] jointly onto the
        L2 ball of radius clip_norm.
        """
        full = np.concatenate([coef.flatten(), intercept.flatten()])
        norm = np.linalg.norm(full)
        if norm > self.clip_norm:
            scale     = self.clip_norm / norm
            coef      = coef      * scale
            intercept = intercept * scale
        return coef, intercept

    def clip_flat(self, flat: np.ndarray) -> np.ndarray:
        """
        Clip a flat 1-D parameter / gradient vector to the L2 ball.
        Used by the MLP path (InvoiceAnomalyNet).
        """
        norm = np.linalg.norm(flat)
        if norm > self.clip_norm:
            flat = flat * (self.clip_norm / norm)
        return flat

    # ── Step 2: Gaussian Noise ────────────────────────────────────────────────

    def add_gaussian_noise(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add calibrated Gaussian noise to (coef, intercept)."""
        noise_std       = self.clip_norm * self.sigma
        noisy_coef      = coef      + np.random.normal(0.0, noise_std, coef.shape)
        noisy_intercept = intercept + np.random.normal(0.0, noise_std, intercept.shape)
        return noisy_coef, noisy_intercept

    def add_gaussian_noise_flat(self, flat: np.ndarray) -> np.ndarray:
        """Add calibrated Gaussian noise to a flat parameter vector (MLP path)."""
        noise_std = self.clip_norm * self.sigma
        return flat + np.random.normal(0.0, noise_std, flat.shape)

    def apply(
        self, coef: np.ndarray, intercept: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Full DP pipeline for (coef, intercept): joint clip → noise."""
        clipped_coef, clipped_intercept = self.clip_parameters(coef.copy(), intercept.copy())
        return self.add_gaussian_noise(clipped_coef, clipped_intercept)

    def apply_flat(self, flat: np.ndarray) -> np.ndarray:
        """Full DP pipeline for a flat gradient/delta vector: clip → noise."""
        return self.add_gaussian_noise_flat(self.clip_flat(flat.copy()))

    # ── Step 3: RDP Epsilon Accounting ────────────────────────────────────────

    def compute_epsilon(self, n_rounds: int) -> float:
        """
        Compute the (ε, δ)-DP guarantee after n_rounds of the subsampled
        Gaussian mechanism.

        SUBSAMPLING AMPLIFICATION (Wang et al. 2019, Mironov 2017 Thm 9):
        When each round samples a fraction q of the population (Poisson
        subsampling), the per-round RDP of the subsampled mechanism is bounded
        by the first-order amplification approximation (tight for small q):

            ε_RDP_sub(α) ≈ q² × α / (2σ²)      per round

        Sequential composition over n_rounds:
            ε_RDP_total(α) = n_rounds × q² × α / (2σ²)

        RDP → (ε, δ)-DP via Balle et al. (2020) Proposition 3:
            ε = ε_RDP_total + log(1 − 1/α)/(α−1) − log(δ)/(α−1)

        Minimised over α ∈ {2, 4, 8, …, 256}.

        With sigma=1.0, sampling_rate=0.03, n_rounds=15, delta=1e-5 → ε ≈ 0.59
        """
        if n_rounds <= 0:
            return 0.0

        q        = self.sampling_rate
        best_eps = float("inf")
        for alpha in [2, 4, 8, 16, 32, 64, 128, 256]:
            eps_rdp    = (q ** 2 * alpha / (2.0 * self.sigma ** 2)) * n_rounds
            conversion = (
                np.log(1.0 - 1.0 / alpha) / (alpha - 1)
                - np.log(self.delta)       / (alpha - 1)
            )
            best_eps = min(best_eps, eps_rdp + conversion)

        return round(float(best_eps), 6)

    def privacy_report(self, n_rounds: int) -> dict:
        eps = self.compute_epsilon(n_rounds)
        return {
            "epsilon":          eps,
            "delta":            self.delta,
            "sigma":            self.sigma,
            "clip_norm":        self.clip_norm,
            "sampling_rate":    self.sampling_rate,
            "rounds_completed": n_rounds,
            "budget_status":    "WITHIN BUDGET" if eps < 1.0 else "EXCEEDED",
            "mechanism":        "Subsampled Gaussian with RDP (Wang et al. 2019 / Balle et al. 2020)",
        }
