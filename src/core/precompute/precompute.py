from __future__ import annotations
from typing import List, Optional
import numpy as np

from numba import cuda as _cuda

from src.core.gpu.kernels import trace_all_kernel
from src.core.gpu.utils import fspl_const, obs_arrays, obs_roughness_array, obs_eps_array
from .static_field import StaticField
from .hash import build_spatial_hash

from src.core.antenna_pattern.ray_dirs_fn import sample_3gpp_dirs_3lobes


def precompute(
    scene,
    seed: Optional[int] = None,
    batch_size: int = 0,
    threads_per_block: int = 256,
    cell_size: Optional[float] = None,
) -> StaticField:
    """
    Trace scene with global ray batching, persistent GPU buffers, and minimal host-device overhead.
    """

    if seed is not None:
        np.random.seed(seed)
    seed_val = int(seed) if seed is not None else 0

    n_max       = int(scene.n_max)
    use_physics = bool(scene.use_physics)
    noise_floor = float(scene.noise_floor_dbm) if use_physics else float('-inf')
    cs          = cell_size if cell_size is not None else 5.0
    fc          = float(scene.transmitters[0].frequency)
    fc_c        = np.float32(fspl_const(fc))

    obs_min_np, obs_max_np = obs_arrays(scene.obstacles)
    obs_rough_np           = obs_roughness_array(scene.obstacles)
    obs_eps_np             = obs_eps_array(scene.obstacles)    
    box_min_np             = np.asarray(scene.box.box_min, dtype=np.float32)
    box_max_np             = np.asarray(scene.box.box_max, dtype=np.float32)


    dirs_per_globe = sample_3gpp_dirs_3lobes(scene.n_rays)
    n_rays_per_tx = sum(d.shape[0] for d in dirs_per_globe)
    N_total = len(scene.transmitters) * n_rays_per_tx

    all_dirs_cpu  = np.empty((N_total, 3), dtype=np.float32)
    all_txpos_cpu = np.empty((N_total, 3), dtype=np.float32)
    all_pwr_cpu   = np.empty((N_total,), dtype=np.float32)
    txid_cpu      = np.empty((N_total,), dtype=np.int32)

    idx = 0
    for tx in scene.transmitters:
        pos_np = np.asarray(tx.position, dtype=np.float32)
        pwr_np = np.float32(tx.tx_power_dbm)
        for i in [0, 1, 2]:
            lobe_rays = dirs_per_globe[i]
            N_lobe = lobe_rays.shape[0]
            end = idx + N_lobe
            
            all_dirs_cpu[idx:end]  = lobe_rays
            all_txpos_cpu[idx:end] = pos_np
            all_pwr_cpu[idx:end]   = pwr_np
            txid_cpu[idx:end]      = tx.tx_id * 3 + i
            
            idx = end

    pos_cpu  = np.empty((n_max + 2, N_total, 3), dtype=np.float32)
    dir_cpu  = np.empty((n_max + 2, N_total, 3), dtype=np.float32)
    sp_cpu   = np.empty((n_max + 2, N_total),    dtype=np.float32)
    npts_cpu = np.empty((N_total,),              dtype=np.int32)


    obs_min_g   = _cuda.to_device(obs_min_np)
    obs_max_g   = _cuda.to_device(obs_max_np)
    obs_rough_g = _cuda.to_device(obs_rough_np)
    obs_eps_g   = _cuda.to_device(obs_eps_np)
    box_min_g   = _cuda.to_device(box_min_np)
    box_max_g   = _cuda.to_device(box_max_np)

    all_dirs_g  = _cuda.to_device(all_dirs_cpu)
    all_txpos_g = _cuda.to_device(all_txpos_cpu)
    all_pwr_g   = _cuda.to_device(all_pwr_cpu)


    _bs = N_total if batch_size <= 0 else batch_size

    pos_g     = _cuda.device_array((n_max + 2, _bs, 3), dtype=np.float32)
    dir_g     = _cuda.device_array((n_max + 2, _bs, 3), dtype=np.float32)
    sp_g      = _cuda.device_array((n_max + 2, _bs),    dtype=np.float32)
    pwr_out_g = _cuda.device_array((_bs,),              dtype=np.float32)
    npts_g    = _cuda.device_array((_bs,),              dtype=np.int32)


    for b_idx, start in enumerate(range(0, N_total, _bs)):
        NB = min(_bs, N_total - start)
        end = start + NB

        seed_off = np.int32(seed_val * 999983 + b_idx * 7919)
        bpg      = (NB + threads_per_block - 1) // threads_per_block

        trace_all_kernel[bpg, threads_per_block](
            pos_g[:, :NB, :], dir_g[:, :NB, :], sp_g[:, :NB], pwr_out_g[:NB], npts_g[:NB],
            all_dirs_g[start:end, :],
            all_txpos_g[start:end, :], 
            all_pwr_g[start:end],
            obs_min_g, obs_max_g,
            obs_rough_g, obs_eps_g,
            box_min_g, box_max_g,
            np.int32(n_max),
            np.float32(noise_floor), fc_c, seed_off,
        )
        

        pos_cpu[:, start:end, :] = pos_g[:, :NB, :].copy_to_host()
        dir_cpu[:, start:end, :] = dir_g[:, :NB, :].copy_to_host()
        sp_cpu[:, start:end]     = sp_g[:, :NB].copy_to_host()
        npts_cpu[start:end]      = npts_g[:NB].copy_to_host()


    sh = build_spatial_hash(pos_cpu, npts_cpu, box_min_np, box_max_np,
                            cs, threads_per_block)

    return StaticField(
        pos_cpu     = pos_cpu,
        dir_cpu     = dir_cpu,
        step_powers = sp_cpu,
        n_pts_cpu   = npts_cpu,
        reached_cpu = np.zeros(N_total, dtype=np.int32),
        tx_ids_cpu  = txid_cpu,
        anchors     = [],
        anchor_ids  = set(),
        spatial_hash= sh,
        fc          = fc,
        scene_ref   = scene,
        rx_ref      = None,
    )