import numpy as np
import math

from geometry import (
    sample_sphere,
    cart_to_sph,
    rotate_dirs
)
from pattern_g3gpp import G_3gPP


def fibonacci_dirs(n: int) -> np.ndarray:
    """Uniformly distributed unit directions on sphere (Fibonacci lattice). Returns (n,3) float32."""
    golden = (1.0 + math.sqrt(5.0)) / 2.0
    i      = np.arange(n, dtype=np.float64)
    theta  = np.arccos(np.clip(1.0 - 2.0 * (i + 0.5) / n, -1.0, 1.0))
    phi    = 2.0 * math.pi * i / golden
    return np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ]).astype(np.float32)




def sample_3gpp_dirs(
    N,
    alpha=1.0,
    batch_size=100_000,
):

    accepted = []

    g_max = 10.0 ** (8.0 / 10.0)

    while sum(len(a) for a in accepted) < N:

        samples = sample_sphere(batch_size)

        theta, phi = cart_to_sph(samples)

        g = G_3gPP(theta, phi) ** alpha


        u = np.random.uniform(
            0.0,
            g_max**alpha,
            batch_size,
        )

        mask = u < g

        accepted.append(samples[mask])

    dirs = np.concatenate(accepted, axis=0)

    return dirs[:N].astype(np.float32)




def sample_3gpp_dirs_3lobes(
    N,
    alpha=2.0,
):
    base = sample_3gpp_dirs(
        N,
        alpha=alpha,
    )

    return [
        rotate_dirs(base, 0),
        rotate_dirs(base, 2*np.pi/3),
        rotate_dirs(base, 4*np.pi/3),
    ]