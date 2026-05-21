import numpy as np

def normalize(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def sph_to_cart(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return np.stack([x, y, z], axis=0)

def cart_to_sph(v):
    """
    v: (N, 3)
    """
    x = v[:, 0]
    y = v[:, 1]
    z = v[:, 2]

    norm = np.linalg.norm(v, axis=1) + 1e-12

    theta = np.arccos(np.clip(z / norm, -1.0, 1.0))
    phi = np.arctan2(y, x)

    return theta, phi


def rotation_matrix_z(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def rotation_matrix_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])


def rotation_matrix_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])


def sample_sphere(M, seed=None):
    if seed is not None:
        np.random.seed(seed)

    phi = np.random.uniform(-np.pi, np.pi, M)
    u   = np.random.uniform(-1.0, 1.0, M)

    theta = np.arccos(u)

    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.column_stack([x, y, z])


def rotate_dirs(
    v: np.ndarray,
    azimuth: float,
    tilt: float = np.deg2rad(20.0),
) -> np.ndarray:

    Ry = rotation_matrix_y(tilt)
    Rz = rotation_matrix_z(azimuth)
    R = Rz @ Ry

    return normalize(v @ R.T).astype(np.float32)