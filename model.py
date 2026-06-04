"""Ramsey equilibrium path calculations for the two-household log/Cobb-Douglas model.

The formulas implement the backward-shooting construction from the TeX file:
inputs: alpha, delta1, delta2, x10, x20.
outputs: K_t, x1_t, x2_t and capital-share diagnostics over a requested finite horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class SimulationInfo:
    alpha: float
    delta1: float
    delta2: float
    x10: float
    x20: float
    K0: float
    initial_share_2: float
    horizon: int
    block_length: int
    exit_time_T: int
    final_exit_ratio_z: Optional[float]
    L: float
    B: float
    zbar: float


def validate_inputs(alpha: float, delta1: float, delta2: float, x10: float, x20: float, horizon: int) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1).")
    if not (0.0 < delta1 < 1.0):
        raise ValueError("delta_1 must be in (0, 1).")
    if not (0.0 < delta2 < delta1):
        raise ValueError("delta_2 must be in (0, delta_1).")
    if x10 < 0.0 or x20 < 0.0:
        raise ValueError("Initial capital holdings x^1_0 and x^2_0 must be nonnegative.")
    if x10 + x20 <= 0.0:
        raise ValueError("At least one initial capital holding must be positive.")
    if horizon < 1:
        raise ValueError("The horizon must be at least 1.")


def constants(alpha: float, delta1: float, delta2: float) -> Tuple[float, float, float]:
    L = (1.0 - alpha) / 2.0
    B = (1.0 + alpha) / 2.0
    zbar = delta1 * (1.0 - alpha) / (delta1 * (1.0 - alpha) + delta2 * (1.0 + alpha))
    return L, B, zbar


def terminal_state(alpha: float, delta1: float, z: float) -> Tuple[float, float, float]:
    """Return (a, b, q) at the final-exit date.

    a = c_T^1 / f(K_{T-1})
    b = c_T^2 / f(K_{T-1})
    q = x_{T-1}^2 / K_{T-1}
    """
    L = (1.0 - alpha) / 2.0
    B = (1.0 + alpha) / 2.0
    q = (z - L) / alpha
    b = z
    a = (1.0 - z) * (B - alpha * delta1) / B
    return a, b, q


def backward_step(alpha: float, delta1: float, delta2: float, a: float, b: float, q: float) -> Tuple[float, float, float, float]:
    """One backward step.

    Given later normalized state (a,b,q), return
    (sigma, A, C, Q), where
      sigma = K_s / f(K_{s-1})
      A     = c_s^1 / f(K_{s-1})
      C     = c_s^2 / f(K_{s-1})
      Q     = x_{s-1}^2 / K_{s-1}
    """
    L = (1.0 - alpha) / 2.0
    D = 1.0 + a / (alpha * delta1) + b / (alpha * delta2)
    sigma = 1.0 / D
    A = sigma * a / (alpha * delta1)
    C = sigma * b / (alpha * delta2)
    Q = (sigma * (b / (alpha * delta2) + q) - L) / alpha
    return sigma, A, C, Q


def q_n(alpha: float, delta1: float, delta2: float, n: int, z: float) -> float:
    """Beginning household-2 share Q_n(z) for an n-period all-active block."""
    if n < 1:
        raise ValueError("n must be at least 1.")
    a, b, q = terminal_state(alpha, delta1, z)
    for _ in range(n - 1):
        _, a, b, q = backward_step(alpha, delta1, delta2, a, b, q)
    return q


def endpoint_rights(alpha: float, delta1: float, delta2: float, max_blocks: int) -> List[float]:
    """Return [Q_0, Q_1(zbar), Q_2(zbar), ...] as right endpoints.

    The zeroth endpoint is 0 = Q_1(L). Entry n is the right endpoint
    Q_n(zbar) of interval I_n.
    """
    L, B, _ = constants(alpha, delta1, delta2)
    a = B - alpha * delta1
    b = L
    q = 0.0
    endpoints = [q]
    for _ in range(max_blocks):
        _, a, b, q = backward_step(alpha, delta1, delta2, a, b, q)
        endpoints.append(q)
    return endpoints


def find_block_and_z(
    alpha: float,
    delta1: float,
    delta2: float,
    target_share: float,
    max_blocks: int = 1000,
    tolerance: float = 1e-12,
) -> Tuple[int, float, List[float]]:
    """Select the active-block length and final-exit ratio for target_share."""
    if not (0.0 < target_share <= 1.0):
        raise ValueError("target_share must be in (0, 1].")

    L, _, zbar = constants(alpha, delta1, delta2)
    endpoints = endpoint_rights(alpha, delta1, delta2, max_blocks)

    n_selected: Optional[int] = None
    for n in range(1, len(endpoints)):
        if target_share <= endpoints[n] + tolerance:
            n_selected = n
            break

    if n_selected is None:
        raise RuntimeError(
            "Could not find a finite active-block length within max_blocks. "
            "Increase max_blocks in the dashboard."
        )

    # Endpoint convention: if target is the right endpoint, z = zbar.
    if abs(target_share - endpoints[n_selected]) <= tolerance:
        return n_selected, zbar, endpoints

    # Otherwise solve Q_n(z) = target_share on [L, zbar] by bisection.
    lo, hi = L, zbar
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        q_mid = q_n(alpha, delta1, delta2, n_selected, mid)
        if q_mid < target_share:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    return n_selected, z, endpoints


def normalized_block(alpha: float, delta1: float, delta2: float, n: int, z: float) -> Dict[str, List[Optional[float]]]:
    """Build normalized active-block arrays.

    Returns one-based arrays for a_t, b_t, sigma_t and zero-based q_t.
    For t = 1,...,n:
      a_t = c_t^1 / f(K_{t-1})
      b_t = c_t^2 / f(K_{t-1})
      sigma_t = K_t / f(K_{t-1})
    For t = 0,...,n-1:
      q_t = x_t^2 / K_t
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    B = (1.0 + alpha) / 2.0
    a: List[Optional[float]] = [None] * (n + 1)
    b: List[Optional[float]] = [None] * (n + 1)
    sigma: List[Optional[float]] = [None] * (n + 1)
    q: List[Optional[float]] = [None] * n

    a[n], b[n], q[n - 1] = terminal_state(alpha, delta1, z)
    sigma[n] = alpha * delta1 * (1.0 - z) / B

    for t in range(n - 1, 0, -1):
        sigma[t], a[t], b[t], q[t - 1] = backward_step(alpha, delta1, delta2, a[t + 1], b[t + 1], q[t])  # type: ignore[arg-type]

    return {"a": a, "b": b, "sigma": sigma, "q": q}


def simulate_ramsey(
    alpha: float,
    delta1: float,
    delta2: float,
    x10: float,
    x20: float,
    horizon: int = 40,
    max_blocks: int = 1000,
) -> Tuple[List[Dict[str, float | int | str | None]], SimulationInfo]:
    """Simulate the unique convergent path over a finite horizon."""
    validate_inputs(alpha, delta1, delta2, x10, x20, horizon)
    K0 = x10 + x20
    share2 = x20 / K0
    L, B, zbar = constants(alpha, delta1, delta2)

    K = [0.0] * (horizon + 1)
    K[0] = K0
    sigma_used: List[Optional[float]] = [None] * (horizon + 1)
    q_path: List[float] = [0.0] * (horizon + 1)
    phase: List[str] = [""] * (horizon + 1)

    if x20 == 0.0:
        block_length = 0
        exit_time_T = 0
        z = None
        for t in range(horizon + 1):
            q_path[t] = 0.0
            phase[t] = "patient continuation"
        for t in range(1, horizon + 1):
            sigma_used[t] = alpha * delta1
            K[t] = sigma_used[t] * (K[t - 1] ** alpha)
    else:
        block_length, z, _ = find_block_and_z(alpha, delta1, delta2, share2, max_blocks=max_blocks)
        exit_time_T = block_length
        block = normalized_block(alpha, delta1, delta2, block_length, z)
        q_block = block["q"]
        sigma_block = block["sigma"]

        for t in range(horizon + 1):
            if t < block_length:
                q_path[t] = float(q_block[t])  # type: ignore[arg-type]
                phase[t] = "active block"
            elif t == block_length:
                q_path[t] = 0.0
                phase[t] = "exit date"
            else:
                q_path[t] = 0.0
                phase[t] = "patient continuation"

        # Match the user's exact initial holdings at t=0; later q's come from the shooting construction.
        q_path[0] = share2

        for t in range(1, horizon + 1):
            if t <= block_length:
                sigma_used[t] = float(sigma_block[t])  # type: ignore[arg-type]
            else:
                sigma_used[t] = alpha * delta1
            K[t] = sigma_used[t] * (K[t - 1] ** alpha)

    rows: List[Dict[str, float | int | str | None]] = []
    previous_share1: Optional[float] = None
    previous_share2: Optional[float] = None

    for t in range(horizon + 1):
        x2 = q_path[t] * K[t]
        x1 = K[t] - x2
        # Pin t=0 exactly to the user-provided initial holdings.
        if t == 0:
            x1, x2 = x10, x20

        share1 = (x1 / K[t]) if K[t] > 0 else None
        share2_t = (x2 / K[t]) if K[t] > 0 else None
        share1_drop = None if previous_share1 is None or share1 is None else previous_share1 - share1
        share2_drop = None if previous_share2 is None or share2_t is None else previous_share2 - share2_t

        rows.append(
            {
                "t": t,
                "K_t": K[t],
                "x1_t": x1,
                "x2_t": x2,
                "x1_t_over_K_t": share1,
                "x2_t_over_K_t": share2_t,
                "x1_share_drop_from_tminus1": share1_drop,
                "x2_share_drop_from_tminus1": share2_drop,
                "saving_ratio_Kt_over_fKtminus1": sigma_used[t],
                "phase": phase[t],
            }
        )

        previous_share1 = share1
        previous_share2 = share2_t

    info = SimulationInfo(
        alpha=alpha,
        delta1=delta1,
        delta2=delta2,
        x10=x10,
        x20=x20,
        K0=K0,
        initial_share_2=share2,
        horizon=horizon,
        block_length=block_length,
        exit_time_T=exit_time_T,
        final_exit_ratio_z=z,
        L=L,
        B=B,
        zbar=zbar,
    )
    return rows, info
