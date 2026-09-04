from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from anndata import AnnData
from pandas import get_dummies
from scanpy import pp


DTP = "float32"
LOADINGS_SEED = 101
NONSPATIAL_SEED = 101
SHUFFLE_SEED = 101
COUNT_SEED = 101


def rescale_spatial_coords(x: np.ndarray, box_side: int = 4) -> np.ndarray:
    xmin = x.min(axis=0)
    x = x - xmin
    x_gmean = np.exp(np.mean(np.log(x.max(axis=0))))
    x = x * (box_side / x_gmean)
    return x - x.mean(axis=0)


def make_grid(n: int, xmin: float = -2, xmax: float = 2, dtype: str = "float32") -> np.ndarray:
    x = np.linspace(xmin, xmax, num=int(np.sqrt(n)), dtype=dtype)
    return np.stack([axis.ravel() for axis in np.meshgrid(x, x)], axis=1)


def sqrt_int(x: int) -> int:
    z = int(round(x**0.5))
    if x == z**2:
        return z
    raise ValueError("x must be a square integer")


def ggblocks(active_factors: tuple[int, ...]) -> np.ndarray:
    patterns = {
        0: [1, 6, 7, 8, 13],
        1: [3, 4, 5, 9, 11, 15, 16, 17],
        2: [18, 24, 25, 30, 31, 32],
        3: [21, 22, 23, 28, 34],
    }
    a = np.zeros((4, 36))
    for factor_index in active_factors:
        a[factor_index, patterns[factor_index]] = 1
    return a


def gen_spatial_factors(nside: int = 36, active_factors: tuple[int, ...] = (0, 1, 2)) -> np.ndarray:
    a = ggblocks(active_factors)
    unit = sqrt_int(a.shape[1])
    if nside % unit != 0:
        raise ValueError("nside must be divisible by the ggblocks unit size")
    ncopy = nside // unit
    n = nside**2
    l = a.shape[0]
    a = a.reshape((l, unit, unit))
    a = np.kron(a, np.ones((1, ncopy, ncopy)))
    return a.reshape((l, n)).T


def gen_spatial_coords(n: int) -> np.ndarray:
    x = make_grid(n)
    x[:, 1] = -x[:, 1]
    return rescale_spatial_coords(x)


def gen_nonspatial_factors(n: int, l: int = 3, nzprob: float = 0.2, seed: int = NONSPATIAL_SEED) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.binomial(1, nzprob, size=(n, l))


def gen_loadings(
    lsp: int,
    lns: int = 3,
    jsp: int = 0,
    jmix: int = 500,
    jns: int = 0,
    expr_mean: float = 10.0,
    mix_frac_spat: float = 0.55,
    seed: int = LOADINGS_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    j = jsp + jmix + jns
    if lsp > 0:
        w = rng.choice(lsp, j, replace=True)
        w_loadings = get_dummies(w).reindex(columns=range(lsp), fill_value=0).to_numpy(dtype=DTP)
    else:
        w_loadings = np.zeros((j, 0))
    if lns > 0:
        v = rng.choice(lns, j, replace=True)
        v_loadings = get_dummies(v).reindex(columns=range(lns), fill_value=0).to_numpy(dtype=DTP)
    else:
        v_loadings = np.zeros((j, 0))

    w_loadings[:jsp, :] *= expr_mean
    v_loadings[:jsp, :] = 0
    w_loadings[jsp : (jsp + jmix), :] *= mix_frac_spat * expr_mean
    v_loadings[jsp : (jsp + jmix), :] *= (1 - mix_frac_spat) * expr_mean
    w_loadings[(jsp + jmix) :, :] = 0
    v_loadings[(jsp + jmix) :, :] *= expr_mean
    return w_loadings, v_loadings


def add_factor_categories(adata: AnnData) -> AnnData:
    one_hot = adata.obsm["spfac"].astype(int).copy()
    category = np.argmax(one_hot, axis=1)
    background = one_hot.sum(axis=1) == 0
    category[background] = 4
    category_names = ["factor 1", "factor 2", "factor 3", "factor 4", "Background"]
    adata.obs["category"] = [category_names[index] for index in category]
    return adata


def sim2anndata(
    locs: np.ndarray,
    outcome: np.ndarray,
    spfac: np.ndarray,
    spload: np.ndarray,
    nsfac: np.ndarray,
    nsload: np.ndarray,
) -> AnnData:
    adata = AnnData(outcome, obsm={"spatial": locs, "spfac": spfac, "nsfac": nsfac}, varm={"spload": spload, "nsload": nsload})
    adata.layers = {"counts": adata.X.copy()}
    pp.log1p(adata)
    idx = list(range(adata.shape[0]))
    random.Random(SHUFFLE_SEED).shuffle(idx)
    return add_factor_categories(adata[idx, :].copy())


def simulate_dataset(
    *,
    active_factors: tuple[int, ...],
    nside: int = 36,
    zero_prob: float,
    count_seed: int = COUNT_SEED,
    jmix: int,
    jsp: int = 0,
    jns: int = 0,
    expr_mean: float = 20.0,
    mix_frac_spat: float = 0.55,
    bkg_mean: float = 0.2,
    nb_shape: float = 10.0,
    nzprob_nsp: float = 0.2,
) -> AnnData:
    spatial_factors = gen_spatial_factors(nside=nside, active_factors=active_factors)
    rng = np.random.default_rng(count_seed)
    n = nside**2
    spatial_coords = gen_spatial_coords(n)
    spatial_loadings, nonspatial_loadings = gen_loadings(
        spatial_factors.shape[1],
        jsp=jsp,
        jmix=jmix,
        jns=jns,
        expr_mean=expr_mean,
        mix_frac_spat=mix_frac_spat,
    )
    nonspatial_factors = gen_nonspatial_factors(n, l=nonspatial_loadings.shape[1], nzprob=nzprob_nsp)
    lam = bkg_mean + spatial_factors @ spatial_loadings.T + nonspatial_factors @ nonspatial_loadings.T
    zero_inflation = rng.binomial(1, zero_prob, size=(n, lam.shape[1]))
    nb_values = rng.negative_binomial(nb_shape, nb_shape / (lam + nb_shape))
    counts = np.where(zero_inflation == 1, 0, nb_values)
    return sim2anndata(spatial_coords, counts, spatial_factors, spatial_loadings, nonspatial_factors, nonspatial_loadings)


def generate_rna_zinb() -> AnnData:
    return simulate_dataset(
        active_factors=(0, 1, 3),
        zero_prob=0.5,
        jmix=1000,
        expr_mean=20.0,
        mix_frac_spat=0.55,
        bkg_mean=0.2,
        nb_shape=10.0,
        nzprob_nsp=0.2,
    )


def generate_adt_100() -> AnnData:
    return simulate_dataset(
        active_factors=(0, 1, 2),
        zero_prob=0.0,
        jmix=100,
        expr_mean=20.0,
        mix_frac_spat=0.55,
        bkg_mean=0.2,
        nb_shape=10.0,
        nzprob_nsp=0.2,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the simulation input h5ad files used in the spaDIVA manuscript.")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rna = generate_rna_zinb()
    adt = generate_adt_100()
    rna.write_h5ad(args.output_dir / "RNA_ZINB.h5ad")
    adt.write_h5ad(args.output_dir / "ADT_100.h5ad")

    print(f"Wrote {args.output_dir / 'RNA_ZINB.h5ad'}")
    print(f"Wrote {args.output_dir / 'ADT_100.h5ad'}")


if __name__ == "__main__":
    main()
