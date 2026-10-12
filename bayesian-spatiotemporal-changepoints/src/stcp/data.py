"""Input validation and geographic/planar spatial lag construction."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


@dataclass(frozen=True)
class Panel:
    cities: list[str]
    years: np.ndarray
    Y: np.ndarray  # city x year including 2000
    xy: np.ndarray
    distance: np.ndarray
    W: np.ndarray
    S: np.ndarray  # city x model year, standardized spatial lag
    spatial_mean: float
    spatial_sd: float
    distance_method: str
    k: int
    t: np.ndarray  # model years, year minus 2000
    Ymodel: np.ndarray


def read_panel(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, encoding="utf-8-sig")
    rename = {"城市": "city", "city_name": "city", "city_id": "city", "rsei_mean": "rsei", "RSEI_mean": "rsei", "RSEI": "rsei"}
    df = df.rename(columns=rename)
    if "city" not in df.columns:
        raise ValueError("RSEI table needs 'city'/'城市'/'city_name' column")
    yearcols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 4 and 1990 <= int(c) <= 2100]
    if yearcols:
        df = df.melt(id_vars=["city"], value_vars=yearcols, var_name="year", value_name="rsei")
    elif not {"year", "rsei"}.issubset(df.columns):
        raise ValueError("RSEI table must be either city,year,rsei or wide with 2000..2021 columns")
    df["city"] = df["city"].astype(str).str.strip()
    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["rsei"] = pd.to_numeric(df["rsei"], errors="raise")
    if df[["city", "year"]].duplicated().any():
        raise ValueError("Duplicate city/year observations")
    if not np.isfinite(df["rsei"]).all() or not df["rsei"].between(0, 1).all():
        raise ValueError("RSEI values must be finite and between 0 and 1")
    return df[["city", "year", "rsei"]]


def read_coords(path: str | Path, distance_method: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"城市": "city", "city_name": "city", "longitude": "lon", "latitude": "lat", "经度": "lon", "纬度": "lat"})
    if "city" not in df:
        raise ValueError("Coordinate table requires city field")
    df["city"] = df["city"].astype(str).str.strip()
    columns = ["lon", "lat"] if distance_method == "haversine" else ["x", "y"]
    if not set(columns).issubset(df.columns):
        raise ValueError(f"Coordinate table needs {columns} for distance={distance_method}")
    df = df[["city"] + columns].copy()
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="raise")
    if not np.isfinite(df[columns].to_numpy()).all():
        raise ValueError("Coordinate values must be finite")
    # Repeated per-year center coordinates are permitted, inconsistent duplicates are not.
    if df.groupby("city")[columns].nunique(dropna=False).to_numpy().max() > 1:
        raise ValueError("Inconsistent coordinate rows for a city")
    df = df.drop_duplicates("city", keep="first").set_index("city")
    if distance_method == "haversine":
        if ((df["lon"].abs() > 180) | (df["lat"].abs() > 90)).any():
            raise ValueError("Longitude/latitude must be decimal degrees")
    return df


def distance_matrix(xy: np.ndarray, method: str) -> np.ndarray:
    if method == "euclidean":
        return cdist(xy, xy, metric="euclidean")
    if method != "haversine":
        raise ValueError("distance must be haversine or euclidean")
    lon = np.deg2rad(xy[:, 0]); lat = np.deg2rad(xy[:, 1]);
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2)**2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2)**2
    return 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # kilometers


def knn_weights(D: np.ndarray, k: int) -> np.ndarray:
    n = D.shape[0]
    if D.shape != (n, n) or not 1 <= k < n:
        raise ValueError("Require square distance matrix and 1 <= k < n_cities")
    if (D[np.triu_indices(n, 1)] <= 0).any():
        raise ValueError("Distinct cities have coincident coordinates")
    D = np.array(D, copy=True)
    np.fill_diagonal(D, np.inf)
    neighbors = np.argsort(D, axis=1, kind="stable")[:, :k]
    W = np.zeros((n, n), dtype=float)
    ii = np.arange(n)[:, None]
    W[ii, neighbors] = 1.0 / D[ii, neighbors]
    W /= W.sum(axis=1, keepdims=True)
    return W


def load_data(panel_path: str, coords_path: str, k: int = 5,
              method: str = "euclidean", first: int = 2000, last: int = 2021) -> Panel:
    d = read_panel(panel_path)
    coords = read_coords(coords_path, method)
    years = np.arange(first, last + 1, dtype=int)
    if len(years) < 8:
        raise ValueError("At least eight years needed")
    p = d.pivot(index="city", columns="year", values="rsei").reindex(columns=years)
    if p.isna().any().any():
        missing = p.isna().stack().loc[lambda x:x].index[:5].tolist()
        raise ValueError(f"Incomplete time series: {missing}")
    # Input order deterministic, and matched cities strictly (unmatched coordinate cities OK).
    cities = sorted(p.index.tolist())
    missing_coords = sorted(set(cities) - set(coords.index))
    if missing_coords:
        raise ValueError(f"Unmatched cities in coordinate table: {missing_coords[:10]}")
    p = p.reindex(cities)
    xy = coords.loc[cities].to_numpy(dtype=float)
    D = distance_matrix(xy, method)
    W = knn_weights(D, k)
    Y = p.to_numpy(float)
    Sraw = W @ Y[:, :-1]
    sm = float(Sraw.mean()); sd = float(Sraw.std(ddof=0))
    if sd <= 1e-12:
        raise ValueError("Spatial lag has zero variation")
    S = (Sraw - sm)/sd
    return Panel(cities, years, Y, xy, D, W, S, sm, sd, method, k,
                 (years[1:] - first).astype(float), Y[:, 1:])
