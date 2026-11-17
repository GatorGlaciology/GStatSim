"""GPU (CUDA) versions of GStatSim's kriging and simulation.

Each function takes a ``backend`` argument:

* ``"gpu"`` (default) runs on an NVIDIA GPU through CuPy. If CuPy or a usable
  CUDA device is missing it raises :class:`BackendUnavailableError`; it never
  falls back to the CPU on its own, so a timing or a result is always from the
  backend you asked for.
* ``"cpu"`` calls GStatSim (``gstatsim.Interpolation``) unchanged, which makes
  it easy to check the GPU result against the reference.

Install the GPU dependency with ``pip install gstatsim[gpu]`` (CUDA 12), or
install the CuPy build that matches your CUDA version.

The GPU results match GStatSim to about 1e-12 with one documented exception:
GStatSim orders neighbors with an unstable sort, so among data points at
exactly the same distance its choice is arbitrary; here ties go to the earlier
data row. On scattered data this never matters. On gridded data a few cells
can pick a different (equally near) neighbor, and in SGS that difference then
carries to later nodes, so compare gridded SGS runs as ensembles.

Public functions
----------------
predict          simple or ordinary kriging (Interpolation.skrige / okrige)
cokrige_mm1      simple cokriging, Markov model 1 (Interpolation.cokrige_mm1)
sgs              one SGS realization (Interpolation.skrige_sgs / okrige_sgs)
realizations     many SGS realizations, simulated side by side on the GPU
cluster_sgs      non-stationary SGS, a variogram per cluster (Interpolation.cluster_sgs)
cosim_mm1        SGS with a secondary variable, Markov model 1 (Interpolation.cosim_mm1)
gpu_status       report whether a GPU can be used, and why not
"""

from __future__ import annotations

import functools
import importlib
import math
import numbers
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import gamma, kv

import gstatsim as gs

__all__ = ['predict', 'cokrige_mm1', 'sgs', 'realizations', 'cluster_sgs', 'cosim_mm1',
           'gpu_status', 'available_backends', 'BackendUnavailableError', 'GpuStatus']


############################

# Backend selection

############################

BACKENDS = ("cpu", "gpu")


class BackendUnavailableError(RuntimeError):
    """Raised when a requested backend cannot run in this environment."""


@dataclass(frozen=True)
class GpuStatus:
    available: bool
    reason: str
    device_name: str | None = None
    compute_capability: str | None = None
    total_memory_bytes: int | None = None
    cuda_runtime_version: int | None = None
    cupy_version: str | None = None


def _import_cupy():
    # Indirection so tests can simulate a missing or broken CuPy install.
    return importlib.import_module("cupy")


def gpu_status() -> GpuStatus:
    """Probe for CuPy and a usable CUDA device without raising."""
    try:
        cp = _import_cupy()
    except ImportError as exc:
        return GpuStatus(
            False,
            f"CuPy is not installed ({exc}). Install it with pip install gstatsim[gpu] "
            "(CUDA 12) or the CuPy build for your CUDA version.",
        )

    try:
        count = cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # CUDARuntimeError, missing driver, etc.
        return GpuStatus(
            False,
            f"CuPy is installed but no CUDA device is usable ({exc}). "
            "Check the NVIDIA driver, or that the job was given a GPU "
            "(e.g. SLURM --gpus=1).",
            cupy_version=cp.__version__,
        )
    if count == 0:
        return GpuStatus(False, "No CUDA devices found.", cupy_version=cp.__version__)

    device = cp.cuda.Device(0)
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode()
    major, minor = device.compute_capability[:-1], device.compute_capability[-1]
    return GpuStatus(
        True,
        "ok",
        device_name=name,
        compute_capability=f"{major}.{minor}",
        total_memory_bytes=int(props["totalGlobalMem"]),
        cuda_runtime_version=int(cp.cuda.runtime.runtimeGetVersion()),
        cupy_version=cp.__version__,
    )


def available_backends() -> tuple:
    """Backends that can run here, CPU first."""
    return ("cpu", "gpu") if gpu_status().available else ("cpu",)


def _resolve_backend(backend: str) -> str:
    """Validate a backend name and confirm it can run; returns the name."""
    if not isinstance(backend, str) or backend.lower() not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    backend = backend.lower()
    if backend == "gpu":
        status = gpu_status()
        if not status.available:
            raise BackendUnavailableError(f"GPU backend requested but unavailable: {status.reason}")
    return backend


############################

# Input checks (shared by both backends, so bad input fails the same way on either)

############################

VTYPES = ("exponential", "gaussian", "spherical", "matern")
METHODS = ("simple", "ordinary")


def _check_prediction_points(points) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"prediction_points must have shape (M, 2), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("prediction_points contains NaN or infinite coordinates")
    return arr


def _check_conditioning_data(df, xx: str, yy: str, zz: str) -> pd.DataFrame:
    """Return a copy with columns renamed to X, Y, Z (GStatSim's convention)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"conditioning_data must be a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in (xx, yy, zz) if c not in df.columns]
    if missing:
        raise ValueError(f"conditioning_data is missing columns {missing}")
    if len(df) == 0:
        raise ValueError("conditioning_data is empty")
    out = df[[xx, yy, zz]].astype(np.float64)
    out.columns = ["X", "Y", "Z"]
    if not np.all(np.isfinite(out.to_numpy())):
        raise ValueError("conditioning_data contains NaN or infinite values; drop or fill them first")
    return out.reset_index(drop=True)


def _check_variogram(vario) -> list:
    """Validate GStatSim's variogram list:
    [azimuth, nugget, major_range, minor_range, sill, vtype(, smoothness for Matern)]
    """
    if not isinstance(vario, (list, tuple)) or len(vario) not in (6, 7):
        raise ValueError(
            "variogram must be [azimuth, nugget, major_range, minor_range, sill, vtype]"
            " plus smoothness s for Matern"
        )
    azimuth, nugget, major, minor, sill, vtype = vario[:6]
    for name, val in (("azimuth", azimuth), ("nugget", nugget), ("major_range", major),
                      ("minor_range", minor), ("sill", sill)):
        if not isinstance(val, numbers.Real) or not np.isfinite(val):
            raise ValueError(f"variogram {name} must be a finite number, got {val!r}")
    if major <= 0 or minor <= 0:
        raise ValueError("variogram ranges must be positive")
    if sill <= 0:
        raise ValueError("variogram sill must be positive")
    if not 0 <= nugget <= sill:
        raise ValueError("variogram nugget must be between 0 and the sill")
    if not isinstance(vtype, str) or vtype.lower() not in VTYPES:
        raise ValueError(f"variogram vtype must be one of {VTYPES}, got {vtype!r}")
    if vtype.lower() == "matern":
        if len(vario) != 7 or not isinstance(vario[6], numbers.Real) or vario[6] <= 0:
            raise ValueError("Matern variogram needs a positive smoothness s as its 7th entry")
    return list(vario)


def _check_search(num_points, radius) -> None:
    # GStatSim's octant search takes num_points // 8 neighbors per octant,
    # so anything below 8 would find no neighbors at all.
    if not isinstance(num_points, numbers.Integral) or num_points < 8:
        raise ValueError(f"num_points must be an integer >= 8, got {num_points!r}")
    if not isinstance(radius, numbers.Real) or not np.isfinite(radius) or radius <= 0:
        raise ValueError(f"radius must be a positive finite number, got {radius!r}")


def _check_method(method: str) -> str:
    if not isinstance(method, str) or method.lower() not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    return method.lower()


def _check_corrcoef(corrcoef) -> float:
    if not isinstance(corrcoef, numbers.Real) or not np.isfinite(corrcoef) or not -1 < corrcoef < 1:
        raise ValueError(f"corrcoef must be a number strictly between -1 and 1, got {corrcoef!r}")
    return float(corrcoef)


def _is_int(v) -> bool:
    return isinstance(v, numbers.Integral) and not isinstance(v, bool)


def _is_seed(v) -> bool:
    return _is_int(v) or isinstance(v, np.random.Generator)


def _gs_seed(v):
    """A seed in the form gstatsim.get_random_generator accepts (it wants a Python int)."""
    return int(v) if _is_int(v) else v


def _seed_list(seed, seeds) -> tuple:
    """Validate seed (one realization) or seeds (several); returns (several?, seed list).

    A seed is an integer, a NumPy Generator, or None, as in GStatSim. A
    Generator is used exactly as GStatSim uses it, so passing the same one to
    several calls (or several times in seeds) gives GStatSim's sequence."""
    if seeds is not None and seed is not None:
        raise ValueError("pass either seed or seeds, not both")
    if seeds is None:
        if seed is not None and not _is_seed(seed):
            raise ValueError(f"seed must be an integer, a NumPy Generator or None, got {seed!r}")
        return False, [_gs_seed(seed)]
    seeds = list(seeds)
    if not seeds or not all(_is_seed(v) for v in seeds):
        raise ValueError("seeds must be a non-empty list of integers or NumPy Generators")
    return True, [_gs_seed(v) for v in seeds]


def _no_matern_on_gpu(vario):
    if vario[5].lower() == "matern":
        raise NotImplementedError("Matern covariance isn't supported on the GPU yet; use backend='cpu'")


############################

# Batched kriging
#
# Written once against the NumPy API: xp=numpy gives a vectorized CPU version,
# xp=cupy runs on the GPU. Each step mirrors a GStatSim function (named in the
# comments) so the two can be checked line by line.

############################

# GStatSim's octant bin edges (NearestNeighbor.nearest_neighbor_search),
# built from the same float expressions so the edges are bit-identical.
OCTANT_EDGES = [-math.pi, -3 * math.pi / 4, -math.pi / 2, -math.pi / 4, 0,
                math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi + 1e-3]

# Elements per (batch x candidates) array in the neighbor search; bounds GPU memory.
_SEARCH_ELEMS = 2 ** 23


def _trim_pool(xp):
    """Release CuPy's cached blocks once they pass half the device memory.

    Batches differ in shape (the number of nearby data changes from batch to
    batch), so CuPy's memory pool keeps caching blocks it can't reuse. Left
    alone, the cache grows by tens of MB per batch until it fills the GPU;
    on Windows the driver then pages GPU memory to system RAM and kriging a
    large grid slows from seconds to hours. Called once per batch.
    """
    if xp is np:
        return
    pool = xp.get_default_memory_pool()
    if pool.total_bytes() > xp.cuda.Device().mem_info[1] // 2:
        pool.free_all_blocks()


def _to_numpy(xp, a):
    return a.get() if xp is not np else a


def _covariance(xp, lag, vario):
    """Covariance.covar."""
    nug, sill, vtype = float(vario[1]), float(vario[4]), vario[5].lower()
    c_struct = sill - nug
    if vtype == "exponential":
        return c_struct * xp.exp(-3 * lag)
    if vtype == "gaussian":
        return c_struct * xp.exp(-3 * xp.square(lag))
    if vtype == "spherical":
        return c_struct * xp.where(lag <= 1.0, 1.0 - 1.5 * lag + 0.5 * xp.power(lag, 3), 0.0)
    if vtype == "matern":
        if xp is not np:
            raise NotImplementedError("Matern covariance needs scipy.special.kv, which CuPy lacks")
        s = float(vario[6])
        scale = 0.45246434 * np.exp(-0.70449189 * s) + 1.7863836
        eff = np.where(lag == 0.0, 1e-8, lag)
        arg = scale * eff * np.sqrt(s)
        c = c_struct * 2 / gamma(s) * np.power(arg, s) * kv(s, 2 * arg)
        return np.where(np.isnan(c), c_struct, c)
    raise ValueError(f"unknown vtype {vario[5]!r}")


def _find_neighbors(xp, px, py, X, Y, radius, num_points):
    """Octant search (NearestNeighbor.nearest_neighbor_search) for a batch of points.

    Returns
    -------
    nbr : (B, 8 * (num_points // 8)) int array
        Row indices into X/Y in GStatSim's layout: octant-major, nearest first
        within an octant; -1 marks an empty slot.
    coincident : (B,) int array
        First data row at exactly the point's location, or -1.
    """
    B, oct_count = len(px), num_points // 8
    nbr = xp.full((B, 8 * oct_count), -1, dtype=xp.int64)
    if len(X) == 0:
        return nbr, xp.full(B, -1, dtype=xp.int64)
    dx = X[None, :] - px[:, None]
    dy = Y[None, :] - py[:, None]
    hit = (dx == 0) & (dy == 0)
    coincident = xp.where(hit.any(axis=1), xp.argmax(hit, axis=1), -1)

    # Squares added as separate arrays (not one expression) so the GPU can't
    # fuse them into an FMA and round differently from NumPy.
    d2 = dx * dx
    d2 += dy * dy
    dist = xp.sqrt(d2)
    angle = xp.arctan2(dx, dy)
    # pd.cut with right-closed bins: octant k holds edges[k] < angle <= edges[k+1].
    octant = xp.searchsorted(xp.asarray(OCTANT_EDGES), angle, side="left") - 1
    valid = (dist < radius) & (octant >= 0)
    del dx, dy, d2, angle

    width = int(valid.sum(axis=1).max()) if B else 0
    if width == 0:
        return nbr, coincident
    order = xp.argsort(xp.where(valid, dist, xp.inf), axis=1, kind="stable")[:, :width]
    oct_sorted = xp.take_along_axis(octant, order, axis=1)
    valid_sorted = xp.take_along_axis(valid, order, axis=1)
    for k in range(8):
        in_k = valid_sorted & (oct_sorted == k)
        rank = xp.cumsum(in_k, axis=1)
        rows, cols = xp.nonzero(in_k & (rank <= oct_count))
        nbr[rows, k * oct_count + rank[rows, cols] - 1] = order[rows, cols]
    return nbr, coincident


def _rotate(x, y, R):
    # coord @ rotation_matrix, written out per component.
    return x * R[0, 0] + y * R[1, 0], x * R[0, 1] + y * R[1, 1]


def _kriging_systems(xp, valid, nx, ny, px, py, vario, R, ordinary):
    """Build the batched kriging matrices (Covariance.make_covariance_matrix/_array).

    nx, ny are the neighbors' coordinates, shape (B, K); `valid` marks real
    neighbors. Empty slots get an identity row/column and a zero right-hand
    side, which gives them zero weight and leaves the real solution unchanged.
    """
    vf = valid.astype(xp.float64)
    m0, m1 = _rotate(nx, ny, R)
    p0, p1 = _rotate(px, py, R)

    d0 = m0[:, :, None] - m0[:, None, :]
    d1 = m1[:, :, None] - m1[:, None, :]
    C = _covariance(xp, xp.sqrt(d0 * d0 + d1 * d1), vario)
    del d0, d1
    C *= vf[:, :, None] * vf[:, None, :]
    K = valid.shape[1]
    diag = xp.arange(K)
    # GStatSim sets the diagonal to the full sill (nugget included).
    C[:, diag, diag] = xp.where(valid, float(vario[4]), 1.0)

    e0, e1 = m0 - p0[:, None], m1 - p1[:, None]
    c0 = _covariance(xp, xp.sqrt(e0 * e0 + e1 * e1), vario) * vf

    if not ordinary:
        return C, c0
    B = len(valid)
    A = xp.zeros((B, K + 1, K + 1))
    A[:, :K, :K] = C
    A[:, K, :K] = vf
    A[:, :K, K] = vf
    b = xp.concatenate([c0, xp.ones((B, 1))], axis=1)
    return A, b


# A covariance matrix whose smallest Cholesky pivot (squared) is below this
# fraction of the sill is treated as numerically singular. A Gaussian
# variogram with no nugget on closely spaced points does this. GStatSim solves
# every system with lstsq, which copes with near-singular matrices; LU or
# Cholesky would return large, meaningless weights instead.
_SINGULAR = 1e-10


_SINGULAR_SGS_MESSAGE = (
    "Some kriging systems are numerically singular (for example a Gaussian variogram with no nugget "
    "on closely spaced points). GStatSim solves these by least squares, which GPU SGS can't do "
    "mid-simulation. Use backend='cpu' for this simulation, or add a small nugget.")


def _nearly_singular(xp, C, sill):
    """Rows of the batch of covariance matrices C that are numerically singular."""
    def one(Ci):
        try:
            return float(np.min(np.diag(np.linalg.cholesky(Ci)) ** 2)) <= _SINGULAR * sill
        except np.linalg.LinAlgError:
            return True

    if xp is np:
        try:
            L = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:  # NumPy raises if any matrix in the batch fails
            return np.array([one(Ci) for Ci in C], dtype=bool)
    else:
        L = xp.linalg.cholesky(C)  # CuPy marks a failed matrix with NaN
    piv = xp.diagonal(L, axis1=1, axis2=2)
    return ~(xp.isfinite(piv).all(axis=1) & ((piv * piv).min(axis=1) > _SINGULAR * sill))


def _solve(xp, A, b, valid, ordinary, sill):
    """Batched solve. Systems that are singular or nearly so are redone with
    lstsq on the CPU, exactly as GStatSim solves every system."""
    try:
        w = xp.linalg.solve(A, b[..., None])[..., 0]
        bad = ~xp.isfinite(w).all(axis=1)
    except np.linalg.LinAlgError:  # NumPy raises; CuPy returns NaN instead
        w = xp.zeros_like(b)
        bad = xp.ones(len(b), dtype=bool)
    K = valid.shape[1]
    bad |= _nearly_singular(xp, A[:, :K, :K], sill)
    if bad.any():
        for i in _to_numpy(xp, xp.nonzero(bad)[0]):
            keep = np.append(_to_numpy(xp, valid[i]), True) if ordinary else _to_numpy(xp, valid[i])
            Ai, bi = _to_numpy(xp, A[i])[np.ix_(keep, keep)], _to_numpy(xp, b[i])[keep]
            wi = np.zeros(len(keep))
            wi[keep] = np.linalg.lstsq(Ai, bi, rcond=None)[0]
            w[i] = xp.asarray(wi)
    return w


def _estimate(xp, w, c0, z, valid, vario, ordinary, global_mean):
    """Estimate and (raw, unclipped) variance, as in skrige/okrige."""
    vf = valid.astype(xp.float64)
    K = valid.shape[1]
    if ordinary:
        w, c0 = w[:, :K], c0[:, :K]
        # Local mean of the neighbors; max(..., 1) only guards rows with none,
        # which are points on a datum and get overwritten by the caller.
        mean = (z * vf).sum(axis=1) / xp.maximum(vf.sum(axis=1), 1)
        centered = z - mean[:, None]
    else:
        mean = global_mean
        centered = z - mean
    est = mean + xp.sum(w * centered * vf, axis=1)
    var = float(vario[4]) - xp.sum(w * c0, axis=1)
    return est, var


def _near_batch(X, Y, p, radius):
    # Only data within `radius` of the batch's bounding box can be neighbors.
    return np.nonzero((X >= p[:, 0].min() - radius) & (X <= p[:, 0].max() + radius) &
                      (Y >= p[:, 1].min() - radius) & (Y <= p[:, 1].max() + radius))[0]


def _batched_krige(xp, points, X, Y, Z, vario, num_points, radius, ordinary, global_mean,
                   batch_size=None):
    """Simple or ordinary kriging of every point (Interpolation.skrige/okrige).

    points, X, Y, Z are NumPy arrays; returns NumPy (estimate, variance).
    """
    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    Xd, Yd, Zd = xp.asarray(X), xp.asarray(Y), xp.asarray(Z)
    est = np.empty(len(points))
    var = np.empty(len(points))
    if batch_size is None:
        batch_size = int(min(4096, max(64, _SEARCH_ELEMS // max(len(X), 1))))

    for start in range(0, len(points), batch_size):
        p = points[start:start + batch_size]
        sub = xp.asarray(_near_batch(X, Y, p, radius))
        Xs, Ys, Zs = Xd[sub], Yd[sub], Zd[sub]
        px, py = xp.asarray(p[:, 0]), xp.asarray(p[:, 1])

        nbr, coincident = _find_neighbors(xp, px, py, Xs, Ys, radius, num_points)
        valid = nbr >= 0
        empty = ~valid.any(axis=1) & (coincident < 0)
        if bool(empty.any()):
            raise ValueError("Unable to find nearest neighbors for some prediction points. "
                             "Try increasing the search radius.")

        j = xp.where(valid, nbr, 0)
        A, b = _kriging_systems(xp, valid, Xs[j], Ys[j], px, py, vario, R, ordinary)
        w = _solve(xp, A, b, valid, ordinary, float(vario[4]))
        z = Zs[j]
        e, v = _estimate(xp, w, b, z, valid, vario, ordinary, global_mean)

        # A point on top of a datum takes its value with zero variance.
        on_datum = coincident >= 0
        e = xp.where(on_datum, Zs[xp.where(on_datum, coincident, 0)], e)
        v = xp.where(on_datum, 0.0, v)
        est[start:start + len(p)] = _to_numpy(xp, e)
        var[start:start + len(p)] = _to_numpy(xp, v)
        _trim_pool(xp)
    return est, var


def _nearest_secondary(xp, points, X2, Y2, batch_size=None):
    """Index of the nearest secondary point to each point
    (NearestNeighbor.nearest_neighbor_secondary: no radius limit, ties to the
    earlier row). Returns a NumPy int array."""
    X2d, Y2d = xp.asarray(X2), xp.asarray(Y2)
    if batch_size is None:
        batch_size = int(min(4096, max(64, _SEARCH_ELEMS // max(len(X2), 1))))
    out = np.empty(len(points), dtype=np.int64)
    for start in range(0, len(points), batch_size):
        p = xp.asarray(points[start:start + batch_size])
        dx = X2d[None, :] - p[:, 0:1]
        dy = Y2d[None, :] - p[:, 1:2]
        d2 = dx * dx
        d2 += dy * dy
        out[start:start + len(p)] = _to_numpy(xp, xp.argmin(d2, axis=1))
        _trim_pool(xp)
    return out


def _mm1_stats(Z1, Z2):
    """Means and variances cokrige_mm1/cosim_mm1 use (np.average, np.var)."""
    return (float(np.average(Z1)), float(np.var(Z1)), float(np.average(Z2)), float(np.var(Z2)))


def _batched_cokrige_mm1(xp, points, X1, Y1, Z1, X2, Y2, Z2, vario, num_points, radius, corrcoef,
                         batch_size=None):
    """Interpolation.cokrige_mm1: simple cokriging with the nearest secondary
    point, Markov model 1. Returns NumPy (estimate, variance)."""
    mean_1, var_1, mean_2, var_2 = _mm1_stats(Z1, Z2)
    # GStatSim replaces the sill with the primary variance (vario[4] = np.var(Z1)).
    vario = list(vario[:4]) + [var_1] + list(vario[5:])
    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    sec = _nearest_secondary(xp, points, X2, Y2)
    X1d, Y1d, Z1d = xp.asarray(X1), xp.asarray(Y1), xp.asarray(Z1)
    est = np.empty(len(points))
    var = np.empty(len(points))
    if batch_size is None:
        batch_size = int(min(4096, max(64, _SEARCH_ELEMS // max(len(X1), 1))))
    rho = float(corrcoef)

    for start in range(0, len(points), batch_size):
        p = points[start:start + batch_size]
        sub = xp.asarray(_near_batch(X1, Y1, p, radius))
        Xs, Ys, Zs = X1d[sub], Y1d[sub], Z1d[sub]
        px, py = xp.asarray(p[:, 0]), xp.asarray(p[:, 1])
        nbr, coincident = _find_neighbors(xp, px, py, Xs, Ys, radius, num_points)
        valid = nbr >= 0
        if bool((~valid.any(axis=1) & (coincident < 0)).any()):
            raise ValueError("Unable to find nearest neighbors for some prediction points. "
                             "Try increasing the search radius.")
        j = xp.where(valid, nbr, 0)
        s = xp.asarray(sec[start:start + len(p)])

        # Primary neighbors plus the secondary point as one extra column.
        K = nbr.shape[1]
        cx = xp.concatenate([Xs[j], xp.asarray(X2)[s][:, None]], axis=1)
        cy = xp.concatenate([Ys[j], xp.asarray(Y2)[s][:, None]], axis=1)
        vall = xp.concatenate([valid, xp.ones((len(p), 1), dtype=bool)], axis=1)
        C, c0 = _kriging_systems(xp, vall, cx, cy, px, py, vario, R, ordinary=False)
        # Markov model 1: secondary covariances are the primary's times rho, unit variance.
        C[:, K, :] *= rho
        C[:, :, K] *= rho
        C[:, K, K] = 1.0
        c0[:, K] *= rho
        w = _solve(xp, C, c0, vall, ordinary=False, sill=var_1)

        z = Zs[j]
        e = (mean_1 + xp.sum(w[:, :K] * (z - mean_1) / np.sqrt(var_1) * valid, axis=1)
             + w[:, K] * (xp.asarray(Z2)[s] - mean_2) / np.sqrt(var_2))
        v = var_1 - xp.sum(w * c0, axis=1)
        on_datum = coincident >= 0
        e = xp.where(on_datum, Zs[xp.where(on_datum, coincident, 0)], e)
        v = xp.where(on_datum, 0.0, v)
        est[start:start + len(p)] = _to_numpy(xp, e)
        var[start:start + len(p)] = _to_numpy(xp, v)
        _trim_pool(xp)
    return est, var


############################

# Sequential Gaussian simulation, CuPy version
#
# SGS is sequential: each node is conditioned on the data *and* every node
# simulated before it, so nodes within one realization can't run in parallel
# without changing the algorithm. Realizations, though, are independent. This
# steps many realizations forward together, one node each per step, and keeps
# all state on the device so the loop never waits on the CPU. Each realization
# is exactly what GStatSim produces with that seed (up to neighbor ties).
#
# Same seed -> same path and same random draws as GStatSim: GStatSim shuffles
# the path, then calls rng.normal(est, sd) once per simulated node, which
# consumes the generator exactly like drawing standard normals up front.

############################

_CHUNK = 2048  # columns added to the conditioning-set slice at a time


def _paths_and_draws(points, X, Y, seeds, copy_datum):
    """Each realization's random path, the datum (if any) each step copies,
    and the standard normal each step uses (the k-th simulated node takes the
    k-th draw), exactly as GStatSim consumes its random generator."""
    S, M = len(seeds), len(points)
    first_datum = {}
    if copy_datum:
        for i, xy in enumerate(zip(X, Y)):
            first_datum.setdefault(xy, i)
    datum_of = np.array([first_datum.get((x, y), -1) for x, y in points])
    node = np.empty((S, M), dtype=np.int64)
    eps = np.zeros((S, M))
    for s, seed in enumerate(seeds):
        rng = gs.get_random_generator(seed)
        path = np.arange(M)
        rng.shuffle(path)
        simulated = datum_of[path] < 0
        # One normal per simulated node, like GStatSim's rng.normal calls. Nodes
        # that copy a datum draw nothing, so a shared Generator ends in the
        # same state as after GStatSim.
        node[s] = path
        eps[s, simulated] = rng.standard_normal(int(simulated.sum()))
    return node, datum_of[node], eps


def _row_neighbors(xp, px, py, X, Y, radius, oct_count, edges):
    """Octant search for B points, each against its own candidate row
    (X, Y have shape (B, n)). Same layout as _find_neighbors, but written so the
    GPU never has to report a size back to the CPU. NaN candidates are ignored."""
    B, n = X.shape
    dx = X - px[:, None]
    dy = Y - py[:, None]
    d2 = dx * dx
    d2 += dy * dy
    dist = xp.sqrt(d2)
    octant = xp.searchsorted(edges, xp.arctan2(dx, dy), side="left") - 1
    valid = (dist < radius) & (octant >= 0)
    octant = xp.where(valid, octant, 8)  # 8 = not a candidate

    # Stable sort by distance, then stable sort by octant: octant-major,
    # nearest first, ties by row order.
    by_dist = xp.argsort(xp.where(valid, dist, xp.inf), axis=1, kind="stable")
    by_oct = xp.argsort(xp.take_along_axis(octant, by_dist, axis=1), axis=1, kind="stable")
    order = xp.take_along_axis(by_dist, by_oct, axis=1)

    counts = (octant[:, :, None] == xp.arange(8)).sum(axis=1)       # (B, 8)
    starts = xp.cumsum(counts, axis=1) - counts
    slot = xp.arange(oct_count)
    pos = xp.minimum(starts[:, :, None] + slot, n - 1).reshape(B, -1)
    picked = xp.take_along_axis(order, pos, axis=1).reshape(B, 8, oct_count)
    return xp.where(slot < counts[:, :, None], picked, -1).reshape(B, -1)


def _xp_sgs_ensemble(xp, points, X, Y, Z, vario, num_points, radius, ordinary, seeds, global_mean):
    """One SGS realization per seed, simulated together.

    points, X, Y, Z are NumPy arrays; returns a NumPy array (len(seeds), M).
    """
    S, M, N = len(seeds), len(points), len(X)
    oct_count = num_points // 8

    # GStatSim copies a datum's value when a node sits exactly on it, but only
    # when the nugget is 0 (it kriges those nodes otherwise).
    node, datum, eps = _paths_and_draws(points, X, Y, seeds, copy_datum=float(vario[1]) == 0)

    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    edges = xp.asarray(OCTANT_EDGES)
    node_d, datum_d, eps_d = xp.asarray(node), xp.asarray(datum), xp.asarray(eps)
    P, Zdata = xp.asarray(points), xp.asarray(Z)

    # Conditioning set per realization: the data, then one column per step.
    # A step that copies a datum adds a NaN placeholder, which never counts as
    # a neighbor, so every realization keeps the same column count.
    # Unfilled columns are NaN too, so the search can run over a slice wider
    # than needed. The slice grows in chunks: growing it by one column per
    # step would give every step new array sizes, which CuPy's memory pool
    # can't reuse, and GPU memory fills up within a few thousand steps.
    Xa, Ya = xp.full((S, N + M), xp.nan), xp.full((S, N + M), xp.nan)
    Za = xp.zeros((S, N + M))
    Xa[:, :N], Ya[:, :N], Za[:, :N] = xp.asarray(X), xp.asarray(Y), Zdata
    out = xp.empty((S, M))
    rows = xp.arange(S)
    no_neighbors = xp.zeros((), dtype=bool)
    singular = xp.zeros((), dtype=bool)

    for i in range(M):
        n = N + i
        width = min(N + M, N + (i // _CHUNK + 1) * _CHUNK)
        Xw, Yw, Zw = Xa[:, :width], Ya[:, :width], Za[:, :width]
        px, py = P[node_d[:, i], 0], P[node_d[:, i], 1]
        nbr = _row_neighbors(xp, px, py, Xw, Yw, radius, oct_count, edges)
        valid = nbr >= 0
        j = xp.where(valid, nbr, 0)
        A, b = _kriging_systems(xp, valid, xp.take_along_axis(Xw, j, axis=1),
                                xp.take_along_axis(Yw, j, axis=1), px, py, vario, R, ordinary)
        w = xp.linalg.solve(A, b[..., None])[..., 0]
        zn = xp.take_along_axis(Zw, j, axis=1)
        est, var = _estimate(xp, w, b, zn, valid, vario, ordinary, global_mean)
        value = est + xp.sqrt(xp.abs(var)) * eps_d[:, i]

        on_datum = datum_d[:, i] >= 0
        no_neighbors |= (~valid.any(axis=1) & ~on_datum).any()
        K = valid.shape[1]
        singular |= (_nearly_singular(xp, A[:, :K, :K], float(vario[4])) & ~on_datum).any()
        value = xp.where(on_datum, Zdata[xp.maximum(datum_d[:, i], 0)], value)
        out[rows, node_d[:, i]] = value
        Xa[:, n] = xp.where(on_datum, xp.nan, px)
        Ya[:, n] = xp.where(on_datum, xp.nan, py)
        Za[:, n] = value

    # Checked once at the end; checking per step would stall the GPU every step.
    if bool(no_neighbors):
        raise ValueError("Unable to find nearest neighbors for some nodes. Try increasing the search radius.")
    if bool(singular):
        raise FloatingPointError(_SINGULAR_SGS_MESSAGE)
    result = _to_numpy(xp, out)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(_SINGULAR_SGS_MESSAGE)
    return result


############################

# Sequential Gaussian simulation, CUDA kernel
#
# The CuPy version above issues ~60 small GPU operations per simulated node;
# for a single realization that launch overhead makes the GPU no faster than
# the CPU. Here one block of threads walks the entire random path itself: for
# each node it computes distances, picks octant neighbors, builds the
# covariance matrix in shared memory, factors it (Cholesky), solves, and draws
# the sample, then moves to the next node. Realizations run as separate blocks
# at the same time. Results are the same as the CuPy version.

############################

_KERNEL_VTYPES = {"exponential": 0, "gaussian": 1, "spherical": 2}

_SOURCE = r"""
#define MAXK __MAXK__
#define SINGULAR __SINGULAR__
#define INF (__longlong_as_double(0x7ff0000000000000ULL))

__constant__ double EDGES[9];

__device__ __forceinline__ double covar(double h, int vtype, double c_struct) {
    // Covariance.covar
    if (vtype == 0) return c_struct * exp(-3.0 * h);
    if (vtype == 1) return c_struct * exp(-3.0 * (h * h));
    return h <= 1.0 ? c_struct * (1.0 - 1.5 * h + 0.5 * (h * h * h)) : 0.0;
}

__device__ __forceinline__ bool before(double d1, int i1, double d2, int i2) {
    // (distance, row) order: nearest first, ties to the earlier row
    return d1 < d2 || (d1 == d2 && i1 < i2);
}

extern "C" __global__ void sgs_kernel(
    double* X, double* Y, double* Z,                 // (S, L) conditioning set, grows as nodes are simulated
    const double* P,                                 // (M, 2) node coordinates
    const long long* node, const long long* datum,   // (S, M) path order; datum to copy (-1 = simulate)
    const double* eps,                               // (S, M) standard normal for each step
    double* dist, int* octv, int* cand,              // (S, L) scratch: candidates within the radius
    double* out, int* flags,                         // (S, M) result; (S) error flags
    const int N, const int M, const int L,
    const double radius, const int oct_count, const int ordinary,
    const double* vparams,          // (C, 8) per variogram: vtype, nugget, sill, R00, R01, R10, R11, -
    int* labels, const int cluster_mode,   // (S, L) cluster of each point (cluster_sgs)
    const double global_mean,
    const double* sec, const int mm1,      // (M, 3) nearest secondary x, y, value (cosim_mm1)
    const double rho, const double var_1, const double mean_2, const double var_2)
{
    const int s = blockIdx.x, tid = threadIdx.x, bs = blockDim.x;
    const int lane = tid & 31, warp = tid >> 5, nwarps = (bs + 31) >> 5;
    X += (size_t)s * L; Y += (size_t)s * L; Z += (size_t)s * L;
    dist += (size_t)s * L; octv += (size_t)s * L; cand += (size_t)s * L;
    labels += (size_t)s * L;
    node += (size_t)s * M; datum += (size_t)s * M; eps += (size_t)s * M; out += (size_t)s * M;

    __shared__ double last_d[8];
    __shared__ int last_i[8];
    __shared__ double red_d[8][32];
    __shared__ int red_i[8][32];
    __shared__ int slot[MAXK], nb[MAXK];
    __shared__ int m_sh, bad_sh, ncand_sh, cl_sh;
    __shared__ double first_d[8];
    __shared__ int first_i[8];
    __shared__ double rx[MAXK], ry[MAXK], zz[MAXK], c0[MAXK], y0[MAXK], y1[MAXK];
    __shared__ double A[MAXK * (MAXK + 1) / 2];   // lower triangle, packed by rows

    const int K = 8 * oct_count;

    for (int i = 0; i < M; ++i) {
        const int z = (int)node[i];
        const double px = P[2 * z], py = P[2 * z + 1];
        const int n = N + i;
        if (datum[i] >= 0) {               // node sits on a datum: copy it
            if (tid == 0) out[z] = Z[datum[i]];
            __syncthreads();
            continue;
        }

        // 1. Distance and octant of every point within the radius
        //    (NearestNeighbor.nearest_neighbor_search), collected into a
        //    candidate list so the selection rounds below skip the rest.
        if (tid == 0) ncand_sh = 0;
        if (tid < 8) { last_d[tid] = -1.0; last_i[tid] = -1; }
        __syncthreads();
        for (int c = tid; c < n; c += bs) {
            const double dx = X[c] - px, dy = Y[c] - py;
            const double d = sqrt(dx * dx + dy * dy);
            if (!(d < radius)) continue;                       // also drops NaN placeholders
            const double a = atan2(dx, dy);
            int o = -1;
            for (int e = 0; e < 9; ++e) o += (EDGES[e] < a);   // pd.cut, right-closed bins
            if (o < 0) continue;
            const int q = atomicAdd(&ncand_sh, 1);
            cand[q] = c; dist[q] = d; octv[q] = o;
        }
        __syncthreads();
        const int ncand = ncand_sh;

        // 2. The oct_count nearest per octant: each round takes the next
        //    candidate after the last one picked, in (distance, row) order.
        for (int r = 0; r < oct_count; ++r) {
            double bd[8]; int bi[8];
            for (int k = 0; k < 8; ++k) { bd[k] = INF; bi[k] = 0x7fffffff; }
            // Candidates are in arbitrary order (atomic append); the
            // (distance, row) comparison makes the result independent of it.
            for (int q = tid; q < ncand; q += bs) {
                const int o = octv[q], c = cand[q];
                const double d = dist[q];
                if (!before(last_d[o], last_i[o], d, c)) continue;
                if (before(d, c, bd[o], bi[o])) { bd[o] = d; bi[o] = c; }
            }
            for (int k = 0; k < 8; ++k) {
                for (int off = 16; off > 0; off >>= 1) {
                    const double od = __shfl_down_sync(0xffffffff, bd[k], off);
                    const int oi = __shfl_down_sync(0xffffffff, bi[k], off);
                    if (before(od, oi, bd[k], bi[k])) { bd[k] = od; bi[k] = oi; }
                }
                if (lane == 0) { red_d[k][warp] = bd[k]; red_i[k][warp] = bi[k]; }
            }
            __syncthreads();
            if (tid < 8) {
                double best_d = INF; int best_i = 0x7fffffff;
                for (int w = 0; w < nwarps; ++w)
                    if (before(red_d[tid][w], red_i[tid][w], best_d, best_i)) {
                        best_d = red_d[tid][w]; best_i = red_i[tid][w];
                    }
                slot[tid * oct_count + r] = best_d < INF ? best_i : -1;
                last_d[tid] = best_d; last_i[tid] = best_i;
                if (r == 0) { first_d[tid] = best_d; first_i[tid] = best_i; }
            }
            __syncthreads();
        }

        // 3. Neighbors in GStatSim's order (octant-major, nearest first)
        if (tid == 0) {
            int m = 0;
            for (int q = 0; q < K; ++q) if (slot[q] >= 0) nb[m++] = slot[q];
            m_sh = m; bad_sh = 0;
            if (m == 0) atomicOr(&flags[s], 1);
            // cluster_sgs: the node takes the cluster of its nearest neighbor
            // (nearest_neighbor_search_cluster), i.e. the best octant's first pick.
            int cl = 0;
            if (cluster_mode && m > 0) {
                int k0 = 0;
                for (int k = 1; k < 8; ++k)
                    if (before(first_d[k], first_i[k], first_d[k0], first_i[k0])) k0 = k;
                cl = labels[first_i[k0]];
            }
            cl_sh = cl;
        }
        __syncthreads();
        const int m = m_sh;
        if (m == 0) { if (tid == 0) out[z] = 0.0; __syncthreads(); continue; }

        // Variogram for this node (the cluster's, or the only one)
        const double* vp = vparams + 8 * cl_sh;
        const int vtype = (int)vp[0];
        const double nug = vp[1], sill = vp[2], c_struct = sill - nug;
        const double R00 = vp[3], R01 = vp[4], R10 = vp[5], R11 = vp[6];

        // 4. Kriging system (Covariance.make_covariance_matrix / _array), rotated coordinates.
        //    cosim_mm1 adds the nearest secondary point as one more row (index m).
        const int mt = m + mm1;
        const double prx = px * R00 + py * R10, pry = px * R01 + py * R11;
        for (int q = tid; q < mt; q += bs) {
            double x, y;
            if (q < m) { const int j = nb[q]; x = X[j]; y = Y[j]; zz[q] = Z[j]; }
            else { x = sec[3 * z]; y = sec[3 * z + 1]; zz[q] = sec[3 * z + 2]; }
            rx[q] = x * R00 + y * R10; ry[q] = x * R01 + y * R11;
            const double e0 = rx[q] - prx, e1 = ry[q] - pry;
            c0[q] = covar(sqrt(e0 * e0 + e1 * e1), vtype, c_struct) * (q < m ? 1.0 : rho);
            y0[q] = c0[q]; y1[q] = 1.0;
        }
        __syncthreads();
        for (int e = tid; e < mt * mt; e += bs) {
            const int a = e / mt, b = e % mt;
            if (b > a) continue;
            double v;
            if (a == b) v = (a < m) ? sill : 1.0;             // Markov model 1: unit secondary variance
            else {
                const double d0 = rx[a] - rx[b], d1 = ry[a] - ry[b];
                v = covar(sqrt(d0 * d0 + d1 * d1), vtype, c_struct) * (a < m ? 1.0 : rho);
            }
            A[a * (a + 1) / 2 + b] = v;
        }
        __syncthreads();

        // 5. Cholesky factorization C = L L^T, in place, one barrier per column:
        //    step k updates the trailing block with column k unscaled (divided
        //    by the pivot) while column k-1, which step k doesn't read, is scaled.
        for (int k = 0; k <= mt; ++k) {
            if (k > 0) {                                   // finish column k-1
                const int kk1 = (k - 1) * k / 2 + (k - 1);
                double p1 = A[kk1];
                if (p1 <= SINGULAR * sill) { if (tid == 0) bad_sh = 1; if (p1 <= 0.0) p1 = 1.0; }
                const double piv = sqrt(p1);
                for (int a = k + tid; a < mt; a += bs) A[a * (a + 1) / 2 + k - 1] /= piv;
                if (tid == 0) A[kk1] = piv;
            }
            if (k < mt) {
                const double p = A[k * (k + 1) / 2 + k];
                const double inv_p = p > 0.0 ? 1.0 / p : 1.0;
                for (int a = k + 1 + warp; a < mt; a += nwarps) {   // one warp per row
                    const int ra = a * (a + 1) / 2;
                    const double lak = A[ra + k] * inv_p;
                    for (int b = k + 1 + lane; b <= a; b += 32) A[ra + b] -= lak * A[b * (b + 1) / 2 + k];
                }
            }
            __syncthreads();
        }

        // 6. Solve C w = c0 (and C v = 1 for ordinary kriging). Little arithmetic
        //    and a dependency at every step, so one warp does it with cheap
        //    warp barriers instead of block-wide ones.
        if (warp == 0) {
            for (int k = 0; k < mt; ++k) {                 // forward: L y = rhs
                const double lkk = A[k * (k + 1) / 2 + k];
                const double v0 = y0[k] / lkk, v1 = y1[k] / lkk;
                __syncwarp();
                if (lane == 0) { y0[k] = v0; y1[k] = v1; }
                for (int a = k + 1 + lane; a < mt; a += 32) {
                    const double l = A[a * (a + 1) / 2 + k];
                    y0[a] -= l * v0; y1[a] -= l * v1;
                }
                __syncwarp();
            }
            for (int k = mt - 1; k >= 0; --k) {            // backward: L^T x = y
                const int row = k * (k + 1) / 2;
                const double v0 = y0[k] / A[row + k], v1 = y1[k] / A[row + k];
                __syncwarp();
                if (lane == 0) { y0[k] = v0; y1[k] = v1; }
                for (int a = lane; a < k; a += 32) { y0[a] -= A[row + a] * v0; y1[a] -= A[row + a] * v1; }
                __syncwarp();
            }
        }
        __syncthreads();

        // 7. Estimate, variance and the simulated value (skrige_sgs / okrige_sgs)
        if (tid == 0) {
            double est, var = sill;
            if (mm1) {
                // cosim_mm1: primary terms scaled by 1/sqrt(var_1) (GStatSim's
                // formula), plus the secondary term; sill = var_1 here.
                est = 0.0;
                for (int q = 0; q < m; ++q) est += y0[q] * (zz[q] - global_mean) / sqrt(var_1);
                est = global_mean + est + y0[m] * (zz[m] - mean_2) / sqrt(var_2);
                for (int q = 0; q < mt; ++q) var -= y0[q] * c0[q];
            } else if (ordinary) {
                // Bordered system via the Schur complement: w = C^-1 c0 - mu C^-1 1, sum(w) = 1
                double s0 = 0.0, s1 = 0.0, mean = 0.0;
                for (int q = 0; q < m; ++q) { s0 += y0[q]; s1 += y1[q]; mean += zz[q]; }
                mean /= m;
                const double mu = (s0 - 1.0) / s1;
                est = 0.0;
                for (int q = 0; q < m; ++q) {
                    const double w = y0[q] - mu * y1[q];
                    est += w * (zz[q] - mean); var -= w * c0[q];
                }
                est += mean;
            } else {
                est = 0.0;
                for (int q = 0; q < m; ++q) { est += y0[q] * (zz[q] - global_mean); var -= y0[q] * c0[q]; }
                est += global_mean;
            }
            const double value = est + sqrt(fabs(var)) * eps[i];
            out[z] = value;
            X[n] = px; Y[n] = py; Z[n] = value; labels[n] = cl_sh;
            if (bad_sh) atomicOr(&flags[s], 2);
        }
        __syncthreads();
    }
}
"""


@functools.lru_cache(maxsize=None)
def _kernel(maxk: int):
    import cupy as cp

    module = cp.RawModule(code=_SOURCE.replace("__MAXK__", str(maxk)).replace("__SINGULAR__", repr(_SINGULAR)),
                          # No fused multiply-add, so distances round exactly like NumPy's.
                          options=("--fmad=false", "-std=c++17"))
    edges = cp.ndarray((9,), cp.float64, module.get_global("EDGES"))
    edges[...] = cp.asarray(OCTANT_EDGES, dtype=cp.float64)
    return module.get_function("sgs_kernel")


# The packed covariance matrix lives in shared memory: 96 neighbors is
# 96*97/2 doubles = 37 KB, inside the 48 KB a block gets by default.
MAX_KERNEL_NEIGHBORS = 96


def _kernel_supports(vario, num_points) -> bool:
    return vario[5].lower() in _KERNEL_VTYPES and 8 <= 8 * (num_points // 8) <= MAX_KERNEL_NEIGHBORS


def _kernel_check(varios, num_points):
    if not all(_kernel_supports(v, num_points) for v in varios):
        raise NotImplementedError(
            "GPU SGS here supports exponential/Gaussian/spherical variograms and up to "
            f"{MAX_KERNEL_NEIGHBORS} neighbors (num_points < {MAX_KERNEL_NEIGHBORS + 8})")


def _kernel_params(vario):
    """Kernel's per-variogram row: vtype, nugget, sill, rotation matrix."""
    R = gs.make_rotation_matrix(vario[0], vario[2], vario[3])
    return [_KERNEL_VTYPES[vario[5].lower()], float(vario[1]), float(vario[4]),
            R[0, 0], R[0, 1], R[1, 0], R[1, 1], 0.0]


def _kernel_run(points, X, Y, Z, labels, varios, num_points, radius, ordinary, seeds, global_mean,
                copy_datum, cluster_mode, threads=256, mm1=None):
    """mm1: None, or (secondary (M, 3) array, rho, var_1, mean_2, var_2) for cosim_mm1."""
    import cupy as cp

    S, M, N = len(seeds), len(points), len(X)
    oct_count = num_points // 8
    node, datum, eps = _paths_and_draws(points, X, Y, seeds, copy_datum)

    L = N + M
    Xa = cp.full((S, L), cp.nan)
    Ya = cp.full((S, L), cp.nan)
    Za = cp.zeros((S, L))
    La = cp.zeros((S, L), dtype=cp.int32)
    Xa[:, :N], Ya[:, :N], Za[:, :N] = cp.asarray(X), cp.asarray(Y), cp.asarray(Z)
    La[:, :N] = cp.asarray(labels, dtype=cp.int32)
    out = cp.empty((S, M))
    flags = cp.zeros(S, dtype=cp.int32)

    K = 8 * oct_count
    # Shared memory must hold K neighbors (+1 secondary point for cosim_mm1). Round
    # up so few variants get compiled, but never past 96 (the 48 KB limit) or
    # the exact size needed.
    maxk = min(32 * -(-K // 32), 96) if mm1 is None else K + 1
    sec, rho, var_1, mean_2, var_2 = mm1 if mm1 is not None else (np.zeros((M, 3)), 0.0, 1.0, 0.0, 1.0)
    _kernel(maxk)(
        (S,), (threads,),
        (Xa, Ya, Za, cp.asarray(np.ascontiguousarray(points, dtype=np.float64)),
         cp.asarray(node), cp.asarray(datum), cp.asarray(eps),
         cp.empty((S, L)), cp.empty((S, L), dtype=cp.int32), cp.empty((S, L), dtype=cp.int32), out, flags,
         np.int32(N), np.int32(M), np.int32(L), np.float64(radius), np.int32(oct_count),
         np.int32(bool(ordinary)), cp.asarray([_kernel_params(v) for v in varios], dtype=cp.float64),
         La, np.int32(bool(cluster_mode)), np.float64(global_mean),
         cp.asarray(np.ascontiguousarray(sec, dtype=np.float64)), np.int32(mm1 is not None),
         np.float64(rho), np.float64(var_1), np.float64(mean_2), np.float64(var_2)),
    )
    f = int(flags.max().get()) if S else 0
    if f & 1:
        raise ValueError("Unable to find nearest neighbors for some nodes. Try increasing the search radius.")
    if f & 2:
        raise FloatingPointError(_SINGULAR_SGS_MESSAGE)
    return out.get()


def _kernel_sgs_ensemble(points, X, Y, Z, vario, num_points, radius, ordinary, seeds, global_mean,
                         threads=256):
    """skrige_sgs / okrige_sgs: one realization per seed (NumPy (len(seeds), M) out)."""
    _kernel_check([vario], num_points)
    # skrige_sgs / okrige_sgs copy a datum under a node only when the nugget is 0.
    return _kernel_run(points, X, Y, Z, np.zeros(len(X)), [vario], num_points, radius, ordinary,
                       seeds, global_mean, copy_datum=float(vario[1]) == 0, cluster_mode=False,
                       threads=threads)


def _gpu_sgs_ensemble(df, points, vario, num_points, radius, method, seeds):
    """CUDA kernel when it supports the settings, otherwise the CuPy version.
    Both give GStatSim's result for each seed (up to neighbor ties)."""
    args = (points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), vario,
            num_points, radius)
    kw = dict(ordinary=(method == "ordinary"), seeds=list(seeds), global_mean=float(df["Z"].mean()))
    if _kernel_supports(vario, num_points):
        return _kernel_sgs_ensemble(*args, **kw)
    import cupy as cp

    return _xp_sgs_ensemble(cp, *args, **kw)


############################

# Public functions

############################

def predict(conditioning_data, prediction_points, variogram, *, num_points, radius,
            method="simple", backend="gpu", xx="X", yy="Y", zz="Z", quiet=True):
    """
    Simple or ordinary kriging at every prediction point
    (GStatSim's Interpolation.skrige / okrige).

    Parameters
    ----------
        conditioning_data : pandas DataFrame
            Data with coordinate columns xx, yy and value column zz. Units must
            match prediction_points and the variogram ranges (e.g. meters)
        prediction_points : numpy.ndarray, shape (M, 2)
            x, y coordinates to estimate, as made by Gridding.prediction_grid
        variogram : list
            [azimuth, nugget, major_range, minor_range, sill, vtype], as in GStatSim,
            plus smoothness s for Matern (Matern runs on the CPU only)
        num_points : int
            Neighbors to use, split evenly across 8 octants (GStatSim's search),
            so the effective maximum is 8 * (num_points // 8)
        radius : int, float
            Neighbor search radius
        method : {'simple', 'ordinary'}
        backend : {'gpu', 'cpu'}
            'gpu' raises BackendUnavailableError if CuPy or a CUDA device is
            missing; it never falls back to the CPU. 'cpu' runs GStatSim itself.
        xx, yy, zz : str
            Column names of the x coordinate, y coordinate and value
        quiet : bool
            Hide GStatSim's progress bar (CPU backend)

    Returns
    -------
        est, var : numpy.ndarray, shape (M,)
            Estimate and kriging variance, in the order of prediction_points.
            Points that coincide with a datum take its value and zero variance.
    """
    method = _check_method(method)
    points = _check_prediction_points(prediction_points)
    df = _check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = _check_variogram(variogram)
    _check_search(num_points, radius)
    backend = _resolve_backend(backend)
    if len(points) == 0:
        return np.empty(0), np.empty(0)

    if backend == "cpu":
        krige = gs.Interpolation.skrige if method == "simple" else gs.Interpolation.okrige
        est, var = krige(points, df, "X", "Y", "Z", num_points, vario, radius, quiet=quiet)
        return np.asarray(est, dtype=np.float64), np.asarray(var, dtype=np.float64)

    _no_matern_on_gpu(vario)
    import cupy as cp

    est, var = _batched_krige(
        cp, points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), vario,
        num_points, radius, ordinary=(method == "ordinary"),
        # Same pandas mean GStatSim uses, so simple kriging's mean is bit-identical.
        global_mean=float(df["Z"].mean()),
    )
    var[var < 0] = 0
    return est, var


def cokrige_mm1(conditioning_data, secondary_data, prediction_points, variogram, *, num_points,
                radius, corrcoef, backend="gpu", xx="X", yy="Y", zz="Z", xx2="X", yy2="Y", zz2="Z",
                quiet=True):
    """
    Simple cokriging with a secondary variable, Markov model 1
    (GStatSim's Interpolation.cokrige_mm1).

    Each point uses its octant neighbors from conditioning_data plus the single
    nearest point of secondary_data (no radius limit), whose covariances are the
    primary ones times corrcoef. As in GStatSim, the variogram's sill is
    replaced by the variance of the primary data; unlike GStatSim, the caller's
    variogram list is left unchanged. The variance is returned unclipped, as
    GStatSim returns it.

    Parameters
    ----------
        conditioning_data, prediction_points, variogram, num_points, radius, backend, quiet
            As in predict
        secondary_data : pandas DataFrame
            Secondary variable, with columns xx2, yy2, zz2
        corrcoef : float
            Correlation coefficient between primary and secondary data, in (-1, 1)

    Returns
    -------
        est, var : numpy.ndarray, shape (M,)
    """
    points = _check_prediction_points(prediction_points)
    df1 = _check_conditioning_data(conditioning_data, xx, yy, zz)
    df2 = _check_conditioning_data(secondary_data, xx2, yy2, zz2)
    vario = _check_variogram(variogram)
    _check_search(num_points, radius)
    corrcoef = _check_corrcoef(corrcoef)
    backend = _resolve_backend(backend)
    if len(points) == 0:
        return np.empty(0), np.empty(0)

    if backend == "cpu":
        est, var = gs.Interpolation.cokrige_mm1(points, df1, "X", "Y", "Z", df2, "X", "Y", "Z",
                                                num_points, list(vario), radius, corrcoef, quiet=quiet)
        return np.asarray(est, dtype=np.float64), np.asarray(var, dtype=np.float64)

    _no_matern_on_gpu(vario)
    import cupy as cp

    return _batched_cokrige_mm1(cp, points, *(df1[c].to_numpy() for c in "XYZ"),
                                *(df2[c].to_numpy() for c in "XYZ"), vario, num_points, radius, corrcoef)


def sgs(conditioning_data, prediction_points, variogram, *, num_points, radius, method="simple",
        seed=None, backend="gpu", xx="X", yy="Y", zz="Z", quiet=True):
    """
    One sequential Gaussian simulation realization
    (GStatSim's Interpolation.skrige_sgs / okrige_sgs).

    Parameters
    ----------
        conditioning_data, prediction_points, variogram, num_points, radius, method, backend, xx, yy, zz, quiet
            As in predict
        seed : int, numpy.random.Generator or None
            As in GStatSim. Same seed and inputs -> same random path and
            random draws on either backend; a Generator shared between calls
            advances exactly as it does in GStatSim. Realizations then agree to ~1e-12 except where
            equidistant neighbors tie; a tie changes that node and, through
            it, later nodes, so compare gridded runs as ensembles.

    Returns
    -------
        sim : numpy.ndarray, shape (M,)
            Simulated values in the units of zz (usually normal scores)

    For several realizations use realizations(), which simulates them side by
    side on the GPU and is much faster per realization.
    """
    method = _check_method(method)
    points = _check_prediction_points(prediction_points)
    df = _check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = _check_variogram(variogram)
    _check_search(num_points, radius)
    _, (seed,) = _seed_list(seed, None)
    backend = _resolve_backend(backend)
    if len(points) == 0:
        return np.empty(0)

    if backend == "cpu":
        f = gs.Interpolation.skrige_sgs if method == "simple" else gs.Interpolation.okrige_sgs
        return np.asarray(f(points, df, "X", "Y", "Z", num_points, vario, radius,
                            seed=seed, quiet=quiet), dtype=np.float64)

    _no_matern_on_gpu(vario)
    return _gpu_sgs_ensemble(df, points, vario, num_points, radius, method, [seed])[0]


def realizations(conditioning_data, prediction_points, variogram, *, seeds, num_points, radius,
                 method="simple", backend="gpu", xx="X", yy="Y", zz="Z", quiet=True):
    """
    One SGS realization per seed.

    Row i equals sgs(..., seed=seeds[i]). On the GPU all realizations are
    simulated together, one node each per step, which is where the speedup
    comes from; on the CPU they run one after another through GStatSim.
    The GPU path is a CUDA kernel that runs each realization as one thread
    block (exponential/Gaussian/spherical, up to 96 neighbors); other settings
    use a slower CuPy version.

    Parameters
    ----------
        seeds : list of int or numpy.random.Generator
            One seed per realization
        conditioning_data, prediction_points, variogram, num_points, radius, method, backend, xx, yy, zz, quiet
            As in predict

    Returns
    -------
        sims : numpy.ndarray, shape (len(seeds), M)
    """
    _, seeds = _seed_list(None, seeds)
    method = _check_method(method)
    points = _check_prediction_points(prediction_points)
    df = _check_conditioning_data(conditioning_data, xx, yy, zz)
    vario = _check_variogram(variogram)
    _check_search(num_points, radius)
    backend = _resolve_backend(backend)
    if len(points) == 0:
        return np.empty((len(seeds), 0))

    if backend == "cpu":
        return np.stack([
            sgs(conditioning_data, prediction_points, variogram, num_points=num_points, radius=radius,
                method=method, seed=s, backend="cpu", xx=xx, yy=yy, zz=zz, quiet=quiet)
            for s in seeds
        ])

    _no_matern_on_gpu(vario)
    return _gpu_sgs_ensemble(df, points, vario, num_points, radius, method, seeds)


def cluster_sgs(conditioning_data, prediction_points, variograms, *, num_points, radius,
                cluster_col="K", seed=None, seeds=None, backend="gpu", xx="X", yy="Y", zz="Z",
                quiet=True):
    """
    Non-stationary SGS with a variogram per cluster (GStatSim's Interpolation.cluster_sgs).
    Simple kriging only, as in GStatSim.

    Parameters
    ----------
        variograms : list of variogram lists, or GStatSim's df_gamma DataFrame
            (column 'Variogram'). Cluster k uses variograms[k]
        cluster_col : str
            Column of conditioning_data with each point's cluster (integers
            0 .. len(variograms) - 1). Each simulated node takes the cluster
            of its nearest neighbor
        seed / seeds : int, or list of int
            One realization for seed (returns shape (M,)), or one per entry of
            seeds (returns shape (len(seeds), M); simulated side by side on the GPU)
        conditioning_data, prediction_points, num_points, radius, backend, xx, yy, zz, quiet
            As in predict
    """
    many, seeds = _seed_list(seed, seeds)
    points = _check_prediction_points(prediction_points)
    df = _check_conditioning_data(conditioning_data, xx, yy, zz)
    _check_search(num_points, radius)
    if hasattr(variograms, "columns"):
        variograms = list(variograms["Variogram"])
    varios = [_check_variogram(v) for v in variograms]
    if cluster_col not in conditioning_data.columns:
        raise ValueError(f"conditioning_data is missing the cluster column {cluster_col!r}")
    labels = np.asarray(conditioning_data[cluster_col], dtype=np.float64)
    if (not np.all(np.isfinite(labels)) or np.any(labels != np.round(labels))
            or labels.min() < 0 or labels.max() >= len(varios)):
        raise ValueError(f"{cluster_col!r} must hold integer cluster ids from 0 to {len(varios) - 1}, "
                         "one per variogram")
    labels = labels.astype(np.int64)
    backend = _resolve_backend(backend)

    if len(points) == 0:
        out = np.empty((len(seeds), 0))
    elif backend == "cpu":
        dfk = df.assign(K=labels)
        gamma_df = pd.DataFrame({"Variogram": varios})
        out = np.stack([
            np.asarray(gs.Interpolation.cluster_sgs(points, dfk, "X", "Y", "Z", "K", num_points, gamma_df,
                                                    radius, seed=s, quiet=quiet), dtype=np.float64)
            for s in seeds])
    else:
        _kernel_check(varios, num_points)
        # cluster_sgs always copies a datum under a node, whatever the nugget,
        # and uses np.average for the simple-kriging mean.
        out = _kernel_run(points, df["X"].to_numpy(), df["Y"].to_numpy(), df["Z"].to_numpy(), labels,
                          varios, num_points, radius, False, seeds,
                          float(np.average(df["Z"].to_numpy())), copy_datum=True, cluster_mode=True)
    return out if many else out[0]


def cosim_mm1(conditioning_data, secondary_data, prediction_points, variogram, *, num_points, radius,
              corrcoef, seed=None, seeds=None, backend="gpu", xx="X", yy="Y", zz="Z", xx2="X",
              yy2="Y", zz2="Z", quiet=True):
    """
    SGS with a secondary variable, Markov model 1 (GStatSim's Interpolation.cosim_mm1).

    Parameters
    ----------
        Same as cokrige_mm1, plus
        seed / seeds : int, or list of int
            One realization for seed (returns shape (M,)), or one per entry of
            seeds (returns shape (len(seeds), M); simulated side by side on the GPU)
    """
    many, seeds = _seed_list(seed, seeds)
    points = _check_prediction_points(prediction_points)
    df1 = _check_conditioning_data(conditioning_data, xx, yy, zz)
    df2 = _check_conditioning_data(secondary_data, xx2, yy2, zz2)
    vario = _check_variogram(variogram)
    _check_search(num_points, radius)
    corrcoef = _check_corrcoef(corrcoef)
    backend = _resolve_backend(backend)

    if len(points) == 0:
        out = np.empty((len(seeds), 0))
    elif backend == "cpu":
        out = np.stack([np.asarray(gs.Interpolation.cosim_mm1(
            points, df1, "X", "Y", "Z", df2, "X", "Y", "Z", num_points, list(vario), radius, corrcoef,
            seed=s, quiet=quiet), dtype=np.float64) for s in seeds])
    else:
        import cupy as cp

        _kernel_check([vario], num_points)
        X1, Y1, Z1 = (df1[c].to_numpy() for c in "XYZ")
        X2, Y2, Z2 = (df2[c].to_numpy() for c in "XYZ")
        mean_1, var_1, mean_2, var_2 = _mm1_stats(Z1, Z2)
        # GStatSim replaces the sill with the primary variance (vario[4] = np.var(Z1)).
        vario = list(vario[:4]) + [var_1] + list(vario[5:])
        # The secondary data never change, so each node's nearest one is found up front.
        j = _nearest_secondary(cp, points, X2, Y2)
        sec = np.column_stack([X2[j], Y2[j], Z2[j]])
        # cosim_mm1 always copies a datum under a node, whatever the nugget.
        out = _kernel_run(points, X1, Y1, Z1, np.zeros(len(X1)), [vario], num_points, radius, False,
                          seeds, mean_1, copy_datum=True, cluster_mode=False,
                          mm1=(sec, float(corrcoef), var_1, mean_2, var_2))
    return out if many else out[0]
