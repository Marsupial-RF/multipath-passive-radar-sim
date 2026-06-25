import numpy as np


def G_3gPP(theta, phi):
    theta_3db = np.deg2rad(65.0)
    phi_3db   = np.deg2rad(65.0)

    SLA_V = 30.0
    A_max = 30.0
    G_max = 8.0

    A_v = -np.minimum(
        12.0 * ((theta - np.pi/2) / theta_3db) ** 2,
        SLA_V
    )

    A_h = -np.minimum(
        12.0 * (phi / phi_3db) ** 2,
        A_max
    )

    A_db = -np.minimum(-(A_v + A_h), A_max) + G_max

    return 10.0 ** (A_db / 10.0)


def gain_from_vectors(vectors, cart_to_sph_fn):
    theta, phi = cart_to_sph_fn(vectors)
    return G_3gPP(theta, phi)
