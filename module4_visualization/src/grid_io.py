"""grid_io.py - 페어 단위로 K3A grid / S2 grid / Simulated 경로를 묶고 RGB 스택을 읽는 헬퍼.

label 은 정합 파이프라인이 부여한 prefix 정수 (예: 10, 11, ...).
- K3A grid: data/interim/grid/{label}_K3A_*_{B|G|R|N}_grid.tif
- S2 grid:  data/interim/grid/{label}_T*_*_B0{2|3|4|8}_grid.tif (B02=Blue, B03=Green, B04=Red, B08=NIR)
- Simulated: data/output/mtf_sim/{label}_K3A_*_simulated_final_2p5m.tif (밴드순 B,G,R,N)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds

from shared.utils.paths import resolve_path

K3A_BAND_SUFFIX: Dict[str, str] = {"B": "Blue", "G": "Green", "R": "Red", "N": "NIR"}
S2_BAND_CODE: Dict[str, str] = {"B02": "Blue", "B03": "Green", "B04": "Red", "B08": "NIR"}
SIMULATED_BAND_INDEX: Dict[str, int] = {"Blue": 1, "Green": 2, "Red": 3, "NIR": 4}


@dataclass
class PairPaths:
    """단일 페어(label) 의 K3A grid / S2 grid / Simulated 경로 묶음."""
    label: int
    k3a_grid: Dict[str, Path]   # {"B","G","R","N"} -> Path
    s2_grid: Dict[str, Path]    # {"B02","B03","B04","B08"} -> Path
    simulated: Optional[Path]   # *_simulated_final_2p5m.tif (없을 수 있음)

    def has_simulated(self) -> bool:
        return self.simulated is not None and self.simulated.exists()


def list_available_labels() -> List[int]:
    """Simulated 출력이 존재하는 label 목록을 정렬해서 반환."""
    mtf_dir = resolve_path("mtf_sim_dir")
    labels: List[int] = []
    for tif in mtf_dir.glob("*_simulated_final_2p5m.tif"):
        try:
            labels.append(int(tif.name.split("_", 1)[0]))
        except ValueError:
            continue
    return sorted(set(labels))


def locate_pair(label: int) -> PairPaths:
    """주어진 label 에 대한 K3A/S2/Simulated 경로를 한 번에 찾아 반환."""
    grid_dir = resolve_path("grid_dir")
    mtf_dir = resolve_path("mtf_sim_dir")
    prefix = f"{label}_"

    k3a_grid: Dict[str, Path] = {}
    for suffix in K3A_BAND_SUFFIX:
        matches = sorted(grid_dir.glob(f"{prefix}K3A_*_{suffix}_grid.tif"))
        if not matches:
            raise FileNotFoundError(f"label={label} K3A {suffix} grid 없음 in {grid_dir}")
        k3a_grid[suffix] = matches[0]

    s2_grid: Dict[str, Path] = {}
    for code in S2_BAND_CODE:
        matches = sorted(grid_dir.glob(f"{prefix}T*_*_{code}_grid.tif"))
        if not matches:
            raise FileNotFoundError(f"label={label} S2 {code} grid 없음 in {grid_dir}")
        s2_grid[code] = matches[0]

    sim_matches = sorted(mtf_dir.glob(f"{prefix}K3A_*_simulated_final_2p5m.tif"))
    simulated = sim_matches[0] if sim_matches else None
    return PairPaths(label=label, k3a_grid=k3a_grid, s2_grid=s2_grid, simulated=simulated)


def _window_from_bbox(src: rasterio.io.DatasetReader, bbox_utm: Optional[Tuple[float, float, float, float]]) -> Optional[Window]:
    if bbox_utm is None:
        return None
    minx, miny, maxx, maxy = bbox_utm
    return from_bounds(minx, miny, maxx, maxy, transform=src.transform).round_lengths().round_offsets()


def _read_band(path: Path, bbox_utm: Optional[Tuple[float, float, float, float]]) -> Tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        window = _window_from_bbox(src, bbox_utm)
        arr = src.read(1, window=window)
        meta = {
            "crs": src.crs,
            "transform": src.window_transform(window) if window is not None else src.transform,
            "res": src.res,
        }
    return arr, meta


def read_rgb_stack(band_paths: Dict[str, Path], rgb_keys: Tuple[str, str, str], bbox_utm: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, dict]:
    """K3A 또는 S2 grid 묶음에서 (R,G,B) 순서로 스택을 읽어 (3,H,W) 반환."""
    bands: List[np.ndarray] = []
    meta: Optional[dict] = None
    for key in rgb_keys:
        arr, m = _read_band(band_paths[key], bbox_utm)
        bands.append(arr)
        if meta is None:
            meta = m
    return np.stack(bands, axis=0), meta or {}


def read_simulated_rgb(simulated_path: Path, bbox_utm: Optional[Tuple[float, float, float, float]] = None) -> Tuple[np.ndarray, dict]:
    """simulated_final_2p5m.tif 에서 R(3)/G(2)/B(1) 순서로 읽어 (3,H,W) 반환."""
    rgb_indices = (SIMULATED_BAND_INDEX["Red"], SIMULATED_BAND_INDEX["Green"], SIMULATED_BAND_INDEX["Blue"])
    with rasterio.open(simulated_path) as src:
        window = _window_from_bbox(src, bbox_utm)
        arrs = [src.read(idx, window=window) for idx in rgb_indices]
        meta = {
            "crs": src.crs,
            "transform": src.window_transform(window) if window is not None else src.transform,
            "res": src.res,
        }
    return np.stack(arrs, axis=0), meta
