"""pipeline.py - K3A↔S2 공간 정합 통합 파이프라인

ortho → 가상격자 생성 → K3A 격자 투영 → S2 격자 리샘플 의 4단계를
하나로 묶어 실행합니다. 결과물은 config/paths.json 의 ortho_dir / grid_dir
(또는 명시적으로 전달된 output_dir) 아래에 저장됩니다.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import rasterio
import rasterio.env

from shared.utils.paths import PROJECT_ROOT as _PROJECT_ROOT, load_paths as _load_paths
from shared.utils.proj_env import PROJ_DATA as _PROJ_DATA

from .grid import create_virtual_grid, reproject_k3a_to_grid, resample_s2_to_grid
from .ortho import orthorectify_k3a_with_rpc

logger = logging.getLogger(__name__)


def run_coregistration_pipeline(
    k3a_tif_path: str | Path,
    s2_tif_path: str | Path,
    dem_path: Optional[str | Path] = None,
    target_crs: str = "EPSG:32652",
    resolution: float = 2.5,
    output_dir: Optional[str | Path] = None,
    pair_label: Optional[str] = None,
) -> Dict[str, Path]:
    """전체 공간 정합 파이프라인을 실행합니다.

    실행 순서:
      1) K3A RPC + DEM 정사보정  → ortho_k3a.tif
      2) 정사보정된 K3A 범위 + S2 범위 → 가상격자 프로필 생성
      3) 정사보정된 K3A → 가상격자 재투영  → k3a_grid.tif
      4) S2 → 가상격자 리샘플링              → s2_grid.tif

    Args:
        k3a_tif_path: 원본 K3A TIF 경로.
        s2_tif_path: S2 GeoTIFF 경로.
        dem_path: DEM GeoTIFF 경로 (None 이면 Z=0).
        target_crs: 목표 좌표계.
        resolution: 가상 격자 해상도 (m).
        output_dir: 결과 저장 디렉토리. None 이면 config 에서 로딩.
        pair_label: 페어 식별자 (예: "1", "2"). 지정하면 두 격자 출력 파일명
            앞에 "{pair_label}_" 가 붙어 K3A·S2 같은 번호로 페어가 표시됨.

    Returns:
        dict with keys:
            - 'ortho_k3a': 정사보정 K3A Path
            - 'k3a_grid': 가상격자 K3A Path
            - 's2_grid': 가상격자 S2 Path
            - 'grid_profile': 가상격자 프로필 dict
    """
    env_kwargs = {"PROJ_LIB": _PROJ_DATA} if _PROJ_DATA else {}
    with rasterio.env.Env(**env_kwargs):
        return _run_pipeline(
            k3a_tif_path=k3a_tif_path,
            s2_tif_path=s2_tif_path,
            dem_path=dem_path,
            target_crs=target_crs,
            resolution=resolution,
            output_dir=output_dir,
            pair_label=pair_label,
        )


def _run_pipeline(
    k3a_tif_path: str | Path,
    s2_tif_path: str | Path,
    dem_path: Optional[str | Path] = None,
    target_crs: str = "EPSG:32652",
    resolution: float = 2.5,
    output_dir: Optional[str | Path] = None,
    pair_label: Optional[str] = None,
) -> Dict[str, Path]:
    paths = _load_paths()

    if output_dir is None:
        ortho_dir = _PROJECT_ROOT / paths.get("ortho_dir", "data/interim/ortho")
        grid_dir = _PROJECT_ROOT / paths.get("grid_dir", "data/interim/grid")
    else:
        output_dir = Path(output_dir)
        ortho_dir = output_dir / "ortho"
        grid_dir = output_dir / "grid"

    k3a_tif_path = Path(k3a_tif_path)
    s2_tif_path = Path(s2_tif_path)

    # ── Step 1: K3A 정사보정 (RPC + DEM) ──
    ortho_k3a_path = ortho_dir / f"{k3a_tif_path.stem}_ortho.tif"
    logger.info("=" * 60)
    logger.info("[Step 1/4] K3A 정사보정 시작")
    ortho_k3a_path = orthorectify_k3a_with_rpc(
        k3a_tif_path=k3a_tif_path,
        out_path=ortho_k3a_path,
        dem_path=dem_path,
        target_crs=target_crs,
        target_resolution=resolution,
    )

    # ── Step 2: 공통 가상격자 생성 (origin = 교집합 좌상단, snap 없음) ──
    logger.info("[Step 2/4] 공통 가상격자 생성 (K3A 2.5m + S2 10m, outer rect 공유)")
    with rasterio.open(s2_tif_path) as s2_src:
        s2_bounds = (s2_src.bounds.left, s2_src.bounds.bottom,
                     s2_src.bounds.right, s2_src.bounds.top)

    grid_profiles = create_virtual_grid(
        ortho_k3a_path=ortho_k3a_path,
        s2_bounds=s2_bounds,
        k3a_resolution=resolution,
        s2_resolution=10.0,
        crs=target_crs,
    )
    k3a_profile = grid_profiles["k3a_profile"]
    s2_profile = grid_profiles["s2_profile"]

    # 페어 라벨이 주어지면 두 출력에 동일 prefix → 디렉토리에서 페어 매칭이 한눈에 보임
    pair_prefix = f"{pair_label}_" if pair_label else ""

    # ── Step 3: 정사보정된 K3A → 가상격자(2.5m) bilinear ──
    logger.info("[Step 3/4] K3A → 가상격자 재투영 (2.5m, Bilinear)")
    k3a_grid_path = grid_dir / f"{pair_prefix}{k3a_tif_path.stem}_grid.tif"
    reproject_k3a_to_grid(
        ortho_k3a_path=ortho_k3a_path,
        grid_profile=k3a_profile,
        out_path=k3a_grid_path,
    )

    # ── Step 4: S2 → 가상격자(10m) bilinear ──
    logger.info("[Step 4/4] S2 → 가상격자 리샘플링 (10m, Bilinear)")
    s2_grid_path = grid_dir / f"{pair_prefix}{s2_tif_path.stem}_grid.tif"
    resample_s2_to_grid(
        s2_tif_path=s2_tif_path,
        grid_profile=s2_profile,
        out_path=s2_grid_path,
    )

    logger.info("=" * 60)
    logger.info("공간 정합 파이프라인 완료!")
    logger.info("  - 정사보정 K3A: %s", ortho_k3a_path)
    logger.info("  - 가상격자 K3A (2.5m): %s", k3a_grid_path)
    logger.info("  - 가상격자 S2  (10m) : %s", s2_grid_path)
    logger.info("=" * 60)

    return {
        "ortho_k3a": ortho_k3a_path,
        "k3a_grid": k3a_grid_path,
        "s2_grid": s2_grid_path,
        "grid_profiles": grid_profiles,
    }
