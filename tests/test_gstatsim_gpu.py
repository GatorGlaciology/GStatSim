"""Tests for gstatsim_gpu against GStatSim itself.

The batched kriging and CuPy-style SGS code also runs on NumPy, so most checks
run without a GPU; tests that need CuPy and a CUDA device skip themselves.
"""

import math

import numpy as np
import pandas as pd
import pytest

import gstatsim as gs
import gstatsim_gpu as gsg

HAVE_GPU = gsg.gpu_status().available
needs_gpu = pytest.mark.skipif(not HAVE_GPU, reason="no usable CUDA GPU")

# Batched code on NumPy, on CuPy, and (SGS only) the CUDA kernel.
XPS = [pytest.param(np, id="numpy")]
if HAVE_GPU:
    import cupy as cp
    XPS.append(pytest.param(cp, id="cupy"))
SGS_IMPLS = XPS + ([pytest.param("cuda", id="cuda-kernel")] if HAVE_GPU else [])

VARIOS = [
    [0, 0.0, 6000.0, 6000.0, 1.0, "Exponential"],
    [30, 0.1, 6000.0, 3000.0, 1.0, "Spherical"],
    [120, 0.05, 5000.0, 2500.0, 0.8, "Gaussian"],
]


def no_cupy():
    raise ImportError("No module named 'cupy'")


@pytest.fixture
def small_field():
    """200 scattered points with a smooth trend plus noise, on a 10 km square."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 10_000, 200)
    y = rng.uniform(0, 10_000, 200)
    z = np.sin(x / 3000) + np.cos(y / 4000) + 0.1 * rng.standard_normal(200)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})
    gx, gy = np.meshgrid(np.arange(250, 10_000, 1000.0), np.arange(250, 10_000, 1000.0))
    pred = np.column_stack([gx.ravel(), gy.ravel()])
    vario = [0, 0.05, 4000.0, 4000.0, 1.0, "Exponential"]
    return df, pred, vario


@pytest.fixture(scope="module")
def scattered():
    rng = np.random.default_rng(1)
    x, y = rng.uniform(0, 2e4, 300), rng.uniform(0, 2e4, 300)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(300)
    pts = np.column_stack([rng.uniform(0, 2e4, 150), rng.uniform(0, 2e4, 150)])
    pts[:3] = np.column_stack([x[:3], y[:3]])  # three points sit on data
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pts


############################

# Backend selection

############################

def test_cpu_always_available():
    assert gsg._resolve_backend("cpu") == "cpu"
    assert gsg._resolve_backend("CPU") == "cpu"
    assert "cpu" in gsg.available_backends()


@pytest.mark.parametrize("bad", ["cuda", "", None, 1])
def test_unknown_backend_rejected(bad):
    with pytest.raises(ValueError, match="backend must be one of"):
        gsg._resolve_backend(bad)


def test_missing_cupy_gives_actionable_error(monkeypatch):
    monkeypatch.setattr(gsg, "_import_cupy", no_cupy)
    assert gsg.available_backends() == ("cpu",)
    with pytest.raises(gsg.BackendUnavailableError, match="CuPy is not installed"):
        gsg._resolve_backend("gpu")


def test_cupy_without_device_gives_actionable_error(monkeypatch):
    class FakeRuntime:
        @staticmethod
        def getDeviceCount():
            raise RuntimeError("cudaErrorNoDevice")

    class FakeCupy:
        __version__ = "0.0-fake"

        class cuda:
            runtime = FakeRuntime

    monkeypatch.setattr(gsg, "_import_cupy", lambda: FakeCupy)
    status = gsg.gpu_status()
    assert not status.available
    assert "no CUDA device is usable" in status.reason
    with pytest.raises(gsg.BackendUnavailableError):
        gsg._resolve_backend("gpu")


def test_gpu_request_never_falls_back_to_cpu(small_field, monkeypatch):
    df, pred, vario = small_field
    monkeypatch.setattr(gsg, "_import_cupy", no_cupy)
    with pytest.raises(gsg.BackendUnavailableError):
        gsg.predict(df, pred, vario, num_points=16, radius=5000)  # backend defaults to "gpu"


############################

# Input checks

############################

GOOD_VARIO = [0, 0.05, 4000.0, 4000.0, 1.0, "Exponential"]


def predict_cpu(df=None, pred=None, vario=None, **kw):
    df = pd.DataFrame({"X": [0.0, 100.0], "Y": [0.0, 100.0], "Z": [1.0, 2.0]}) if df is None else df
    pred = np.array([[50.0, 50.0]]) if pred is None else pred
    kw.setdefault("num_points", 8)
    kw.setdefault("radius", 1000.0)
    kw.setdefault("backend", "cpu")
    return gsg.predict(df, pred, GOOD_VARIO if vario is None else vario, **kw)


@pytest.mark.parametrize("pred, msg", [
    (np.zeros((3, 3)), "shape"),
    (np.zeros(4), "shape"),
    (np.array([[np.nan, 0.0]]), "NaN"),
])
def test_bad_prediction_points(pred, msg):
    with pytest.raises(ValueError, match=msg):
        predict_cpu(pred=pred)


@pytest.mark.parametrize("df, err, msg", [
    (np.zeros((2, 3)), TypeError, "DataFrame"),
    (pd.DataFrame({"X": [0.0], "Y": [0.0]}), ValueError, "missing columns"),
    (pd.DataFrame({"X": [], "Y": [], "Z": []}), ValueError, "empty"),
    (pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [np.nan]}), ValueError, "NaN"),
])
def test_bad_conditioning_data(df, err, msg):
    with pytest.raises(err, match=msg):
        predict_cpu(df=df)


@pytest.mark.parametrize("vario, msg", [
    ([0, 0, 1, 1, 1], "variogram must be"),
    ([0, 0, -1, 1, 1, "Exponential"], "ranges must be positive"),
    ([0, 0, 1, 1, 0, "Exponential"], "sill must be positive"),
    ([0, 2, 1, 1, 1, "Exponential"], "nugget"),
    ([0, 0, 1, 1, 1, "Cubic"], "vtype"),
    ([0, 0, 1, 1, 1, "Matern"], "smoothness"),
    ([0, 0, float("nan"), 1, 1, "Exponential"], "finite"),
])
def test_bad_variogram(vario, msg):
    with pytest.raises(ValueError, match=msg):
        predict_cpu(vario=vario)


@pytest.mark.parametrize("kw, msg", [
    ({"num_points": 7}, "num_points"),
    ({"num_points": 8.0}, "num_points"),
    ({"radius": 0}, "radius"),
    ({"radius": float("inf")}, "radius"),
    ({"method": "universal"}, "method"),
])
def test_bad_search_and_method(kw, msg):
    with pytest.raises(ValueError, match=msg):
        predict_cpu(**kw)


def test_bad_input_fails_before_backend_check(monkeypatch):
    # A bad input gives the same error whether or not a GPU is present.
    monkeypatch.setattr(gsg, "_import_cupy", no_cupy)
    with pytest.raises(ValueError, match="num_points"):
        predict_cpu(num_points=7, backend="gpu")


def test_matern_rejected_on_gpu(monkeypatch):
    # Pretend a GPU is present so the call reaches the GPU path.
    monkeypatch.setattr(gsg, "_resolve_backend", lambda b: b)
    with pytest.raises(NotImplementedError, match="Matern"):
        predict_cpu(vario=[0, 0, 1000.0, 1000.0, 1.0, "Matern", 1.5], backend="gpu")


def test_bad_seed_rejected(small_field):
    df, pred, vario = small_field
    with pytest.raises(ValueError, match="seed"):
        gsg.sgs(df, pred, vario, num_points=8, radius=1e3, seed=1.5, backend="cpu")


############################

# CPU backend is GStatSim

############################

@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_cpu_backend_matches_gstatsim_exactly(small_field, method):
    df, pred, vario = small_field
    est, var = gsg.predict(df, pred, vario, num_points=16, radius=5000, method=method, backend="cpu")
    ref = gs.Interpolation.skrige if method == "simple" else gs.Interpolation.okrige
    ref_est, ref_var = ref(pred, df.copy(), "X", "Y", "Z", 16, vario, 5000, quiet=True)
    np.testing.assert_array_equal(est, ref_est)
    np.testing.assert_array_equal(var, ref_var)


def test_custom_column_names_and_input_left_unchanged(small_field):
    df, pred, vario = small_field
    renamed = df.rename(columns={"X": "east", "Y": "north", "Z": "bed"})
    before = renamed.copy()
    est, _ = gsg.predict(renamed, pred, vario, num_points=16, radius=5000,
                         xx="east", yy="north", zz="bed", backend="cpu")
    ref, _ = gsg.predict(df, pred, vario, num_points=16, radius=5000, backend="cpu")
    np.testing.assert_array_equal(est, ref)
    pd.testing.assert_frame_equal(renamed, before)


def test_empty_prediction_points(small_field):
    df, _, vario = small_field
    est, var = gsg.predict(df, np.empty((0, 2)), vario, num_points=16, radius=5000, backend="cpu")
    assert est.shape == var.shape == (0,)


############################

# Batched kriging (NumPy and GPU) against GStatSim

############################

def batched_krige(xp, points, df, vario, k, radius, ordinary):
    est, var = gsg._batched_krige(xp, points, df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy(),
                                  vario, k, radius, ordinary, float(df.Z.mean()))
    return est, np.where(var < 0, 0, var)


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("vario", VARIOS, ids=lambda v: f"{v[5]}-nug{v[1]}")
@pytest.mark.parametrize("ordinary", [False, True], ids=["SK", "OK"])
def test_scattered_kriging_matches_gstatsim(xp, vario, ordinary, scattered):
    """No distance ties, so every cell must agree to 1e-6."""
    df, pts = scattered
    f = gs.Interpolation.okrige if ordinary else gs.Interpolation.skrige
    ref_est, ref_var = f(pts, df.copy(), "X", "Y", "Z", 32, vario, 8000, quiet=True)
    est, var = batched_krige(xp, pts, df, vario, 32, 8000, ordinary)
    np.testing.assert_allclose(est, ref_est, rtol=0, atol=1e-6)
    np.testing.assert_allclose(var, ref_var, rtol=0, atol=1e-6)


@pytest.mark.parametrize("xp", XPS)
def test_ties_go_to_earlier_row(xp):
    # Four data points at the same distance in octant 4 (0 < angle <= pi/4);
    # only one slot per octant, so row order must decide.
    r = 1000.0
    angles = [0.1, 0.3, 0.5, 0.7]
    X = np.array([r * math.sin(a) for a in angles])
    Y = np.array([r * math.cos(a) for a in angles])
    nbr, _ = gsg._find_neighbors(xp, xp.zeros(1), xp.zeros(1), xp.asarray(X), xp.asarray(Y), 5000, 8)
    picked = [int(j) for j in gsg._to_numpy(xp, nbr[0]) if j >= 0]
    dist = np.hypot(X, Y)
    assert picked == [int(np.flatnonzero(dist == dist.min())[0])]


@pytest.mark.parametrize("xp", XPS)
def test_gridded_differences_are_only_equidistant_swaps(xp):
    """On gridded data a cell may differ from GStatSim only when equidistant
    points swap places in the neighbor set (GStatSim's tie order is arbitrary)."""
    rng = np.random.default_rng(5)
    gx, gy = np.meshgrid(np.arange(0, 30_000, 1000.0), np.arange(0, 30_000, 1000.0))
    keep = rng.random(gx.size) < 0.4  # sparse, survey-like coverage
    x, y = gx.ravel()[keep], gy.ravel()[keep]
    df = pd.DataFrame({"X": x, "Y": y, "Z": np.sin(x / 7e3) + np.cos(y / 9e3)})
    pts = np.column_stack([gx.ravel(), gy.ravel()])[::7]
    vario = VARIOS[0]
    ref_est, _ = gs.Interpolation.skrige(pts, df.copy(), "X", "Y", "Z", 16, vario, 6000, quiet=True)
    est, _ = batched_krige(xp, pts, df, vario, 16, 6000, False)

    differs = np.flatnonzero(np.abs(est - ref_est) > 1e-6)
    nbr, _ = gsg._find_neighbors(np, pts[differs, 0], pts[differs, 1], x, y, 6000, 16)
    for i, row in zip(differs, nbr):
        ours = {(x[j], y[j]) for j in row if j >= 0}
        theirs = {(a, b) for a, b, _ in gs.NearestNeighbor.nearest_neighbor_search(6000, 16, pts[i], df)}
        assert ours != theirs, f"cell {i}: same neighbors but values differ"

        def dists(s, p=pts[i]):
            return sorted(round(math.hypot(a - p[0], b - p[1]), 6) for a, b in s)
        assert dists(ours - theirs) == dists(theirs - ours), f"cell {i}: neighbor sets differ beyond ties"


@pytest.mark.parametrize("xp", XPS)
def test_no_neighbors_raises(xp):
    df = pd.DataFrame({"X": [0.0], "Y": [0.0], "Z": [1.0]})
    with pytest.raises(ValueError, match="search radius"):
        batched_krige(xp, np.array([[1e6, 1e6]]), df, VARIOS[0], 8, 1000, False)


@needs_gpu
@pytest.mark.parametrize("method", ["simple", "ordinary"])
def test_predict_gpu_matches_cpu(method, scattered):
    df, pts = scattered
    cpu = gsg.predict(df, pts, VARIOS[1], num_points=32, radius=8000, method=method, backend="cpu")
    gpu = gsg.predict(df, pts, VARIOS[1], num_points=32, radius=8000, method=method, backend="gpu")
    np.testing.assert_allclose(gpu[0], cpu[0], rtol=0, atol=1e-6)
    np.testing.assert_allclose(gpu[1], cpu[1], rtol=0, atol=1e-6)


@pytest.fixture(scope="module")
def with_secondary():
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 3e4, 300), rng.uniform(0, 3e4, 300)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(300)
    g = np.arange(500, 30000, 1000.0)
    gx, gy = np.meshgrid(g, g)
    x2, y2 = gx.ravel() + 3.7, gy.ravel() - 11.1   # offset so no point coincides with the grid
    z2 = np.sin(x2 / 5e3) + 0.5 * np.cos(y2 / 9e3) + 0.3 * rng.standard_normal(len(x2))
    pts = np.column_stack([rng.uniform(0, 3e4, 150), rng.uniform(0, 3e4, 150)])
    pts[:2] = np.column_stack([x[:2], y[:2]])
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pd.DataFrame({"X": x2, "Y": y2, "Z": z2}), pts


@pytest.mark.parametrize("xp", XPS)
@pytest.mark.parametrize("vario", VARIOS[:2], ids=lambda v: v[5])
def test_cokrige_mm1_matches_gstatsim(xp, vario, with_secondary):
    df1, df2, pts = with_secondary
    ref_est, ref_var = gs.Interpolation.cokrige_mm1(pts, df1.copy(), "X", "Y", "Z", df2.copy(), "X", "Y", "Z",
                                                    32, list(vario), 9000, 0.6, quiet=True)
    est, var = gsg._batched_cokrige_mm1(xp, pts, *(df1[c].to_numpy() for c in "XYZ"),
                                        *(df2[c].to_numpy() for c in "XYZ"), vario, 32, 9000, 0.6)
    np.testing.assert_allclose(est, ref_est, rtol=0, atol=1e-6)
    np.testing.assert_allclose(var, ref_var, rtol=0, atol=1e-6)


def test_corrcoef_validated(with_secondary):
    df1, df2, pts = with_secondary
    with pytest.raises(ValueError, match="corrcoef"):
        gsg.cokrige_mm1(df1, df2, pts, VARIOS[0], num_points=16, radius=5000, corrcoef=1.0, backend="cpu")


############################

# SGS (NumPy, CuPy and the CUDA kernel) against GStatSim with a fixed seed

############################

@pytest.fixture(scope="module")
def field():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 2e4, 200), rng.uniform(0, 2e4, 200)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(200)
    pts = np.column_stack([rng.uniform(0, 2e4, 120), rng.uniform(0, 2e4, 120)])
    return pd.DataFrame({"X": x, "Y": y, "Z": z}), pts


def sgs_ensemble(impl, df, pts, vario, method, seeds):
    args = (pts, df.X.to_numpy(), df.Y.to_numpy(), df.Z.to_numpy(), vario, 16, 8000,
            method == "ordinary", seeds, float(df.Z.mean()))
    if impl == "cuda":
        return gsg._kernel_sgs_ensemble(*args)
    return gsg._xp_sgs_ensemble(impl, *args)


def sgs_cpu(df, pts, vario, method="simple", seed=7):
    return gsg.sgs(df, pts, vario, num_points=16, radius=8000, method=method, seed=seed, backend="cpu")


@pytest.mark.parametrize("impl", SGS_IMPLS)
@pytest.mark.parametrize("method", ["simple", "ordinary"])
@pytest.mark.parametrize("vario", [
    [20, 0.0, 6000.0, 3000.0, 1.0, "Exponential"],
    [0, 0.1, 6000.0, 6000.0, 1.0, "Spherical"],
], ids=["exp-nug0", "sph-nug0.1"])
def test_sgs_fixed_seed_matches_gstatsim(impl, method, vario, field):
    df, pts = field
    pts = pts.copy()
    if vario[1] == 0:
        # Nodes on data take the datum's value (nugget 0 only).
        pts[:2] = df[["X", "Y"]].to_numpy()[:2]
    # With nugget > 0, GStatSim simulates a node sitting on a datum and adds it
    # to the conditioning set, creating two points at one location; those
    # always tie, so that case can't be compared node by node.
    ref = sgs_cpu(df, pts, vario, method)
    np.testing.assert_allclose(sgs_ensemble(impl, df, pts, vario, method, [7])[0], ref, rtol=0, atol=1e-6)


@pytest.mark.parametrize("impl", SGS_IMPLS)
def test_sgs_ensemble_rows_match_gstatsim_per_seed(impl, field):
    """Stepping realizations together must not change any of them."""
    df, pts = field
    pts = pts.copy()
    pts[:3] = df[["X", "Y"]].to_numpy()[:3]  # some steps copy a datum, others don't
    vario = [20, 0.0, 6000.0, 3000.0, 1.0, "Exponential"]
    seeds = [3, 11, 42]
    ens = sgs_ensemble(impl, df, pts, vario, "simple", seeds)
    for row, seed in zip(ens, seeds):
        np.testing.assert_allclose(row, sgs_cpu(df, pts, vario, seed=seed), rtol=0, atol=1e-6)


@pytest.mark.parametrize("impl", SGS_IMPLS)
def test_seed_controls_realization(impl, field):
    df, pts = field
    vario = [0, 0.0, 6000.0, 6000.0, 1.0, "Exponential"]
    a, b, c = (sgs_ensemble(impl, df, pts, vario, "simple", [s])[0] for s in (1, 1, 2))
    np.testing.assert_array_equal(a, b)
    assert not np.allclose(a, c)


@needs_gpu
def test_realizations_rows_equal_sgs(field):
    df, pts = field
    vario = [20, 0.0, 6000.0, 3000.0, 1.0, "Exponential"]
    kw = dict(num_points=16, radius=8000)
    ens = gsg.realizations(df, pts, vario, seeds=[4, 5], **kw)
    for row, seed in zip(ens, (4, 5)):
        np.testing.assert_allclose(row, gsg.sgs(df, pts, vario, seed=seed, **kw), rtol=0, atol=1e-12)


@pytest.fixture(scope="module")
def clustered():
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 3e4, 300), rng.uniform(0, 3e4, 300)
    z = np.sin(x / 5e3) + np.cos(y / 7e3) + 0.2 * rng.standard_normal(300)
    k = (x > 1.5e4).astype(int) + 2 * (y > 2e4).astype(int)  # four quadrant-like clusters
    pts = np.column_stack([rng.uniform(0, 3e4, 150), rng.uniform(0, 3e4, 150)])
    pts[:3] = np.column_stack([x[:3], y[:3]])  # nodes on data copy the datum
    varios = [[0, 0.0, 6000.0, 6000.0, 1.0, "Exponential"], [45, 0.1, 8000.0, 3000.0, 0.8, "Spherical"],
              [90, 0.0, 5000.0, 2500.0, 1.0, "Gaussian"], [0, 0.05, 4000.0, 4000.0, 1.2, "Exponential"]]
    return pd.DataFrame({"X": x, "Y": y, "Z": z, "K": k}), pts, varios


@needs_gpu
def test_cluster_sgs_matches_gstatsim(clustered):
    df, pts, varios = clustered
    kw = dict(num_points=24, radius=9000)
    ens = gsg.cluster_sgs(df, pts, varios, seeds=[7, 8], backend="gpu", **kw)
    for row, seed in zip(ens, (7, 8)):
        ref = gsg.cluster_sgs(df, pts, varios, seed=seed, backend="cpu", **kw)
        np.testing.assert_allclose(row, ref, rtol=0, atol=1e-6)


def test_cluster_labels_validated(clustered):
    df, pts, varios = clustered
    kw = dict(num_points=24, radius=9000, backend="cpu")
    with pytest.raises(ValueError, match="cluster ids"):
        gsg.cluster_sgs(df.assign(K=df.K + 1), pts, varios, **kw)
    with pytest.raises(ValueError, match="missing the cluster column"):
        gsg.cluster_sgs(df.drop(columns="K"), pts, varios, **kw)


@needs_gpu
def test_cosim_mm1_matches_gstatsim_and_leaves_vario_alone(with_secondary):
    df1, df2, pts = with_secondary
    vario = [30, 0.1, 8000.0, 4000.0, 1.0, "Spherical"]
    kw = dict(num_points=32, radius=9000, corrcoef=0.6)
    gpu = gsg.cosim_mm1(df1, df2, pts, vario, seeds=[7, 8], backend="gpu", **kw)
    for row, seed in zip(gpu, (7, 8)):
        np.testing.assert_allclose(row, gsg.cosim_mm1(df1, df2, pts, vario, seed=seed, backend="cpu", **kw),
                                   rtol=0, atol=1e-6)
    assert vario[4] == 1.0  # GStatSim itself would have overwritten the sill
