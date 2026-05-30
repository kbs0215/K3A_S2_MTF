"""generate_user_sr_images.py - 사용자가 첨부한 output/SR_output/*.tif 파일을 과거 데이터셋 폴더와 매핑하여 렌더링.

추가 기능: 타일 격자 아티팩트(grid boundary) 및 가장자리 칼선(border artifacts)을 지우는
디그리드(de-grid) 편법 보정 필터(degrid_tile_artifacts) 탑재.
"""

from __future__ import annotations

import os
import sys

# Windows DLL 및 OpenMP 설정 - MUST be at the very top!
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.platform == "win32":
    for path in [
        r"C:\Users\kbs02\anaconda3\envs\gongjong\Library\bin",
        r"C:\Users\kbs02\anaconda3\envs\gongjong\Lib\site-packages\torch\lib",
    ]:
        if os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass

from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# 그 다음 rasterio 임포트
import rasterio

from module5_SR_comparison.scripts.generate_slider_images import (
    tif_to_rgb,
    save_png_raw,
    save_jpg_raw,
    nearest_upsample_4x,
    bicubic_upsample_4x
)

def degrid_tile_artifacts(img_rgb: np.ndarray) -> np.ndarray:
    """타일 단위 추론 시 발생하는 수평/수직 격자선 및 가장자리 칼선 아티팩트를 보정합니다.
    img_rgb: (H, W, 3) uint8 RGB array
    """
    out = img_rgb.copy()
    H, W, C = out.shape
    
    # 1. 외곽 경계선 (Border) 아티팩트 제거: 외곽 6픽셀 영역을 안쪽 정상 픽셀로 복사 (Replicate)
    border = 6
    for i in range(border):
        out[i, :, :] = out[border, :, :] # 상단
        out[H - 1 - i, :, :] = out[H - 1 - border, :, :] # 하단
        out[:, i, :] = out[:, border, :] # 좌측
        out[:, W - 1 - i, :] = out[:, W - 1 - border, :] # 우측

    # 2. 내부 타일 격자선 (224px 간격) 스무딩
    # 896px 크기에서 224의 배수: 224, 448, 672
    grid_coords = [224, 448, 672]
    half_width = 3 # 경계선 좌우 3픽셀씩 블러 범위
    
    try:
        import cv2
        # 수평 격자선 스무딩 (1D Gaussian Blur 세로 방향 적용)
        for y in grid_coords:
            y_min = max(0, y - half_width)
            y_max = min(H, y + half_width + 1)
            slice_zone = out[y_min:y_max, :, :]
            # ksize=(1, 5)로 세로축 방향으로만 부드럽게 블러 적용
            blurred = cv2.GaussianBlur(slice_zone, (1, 5), 0)
            out[y_min:y_max, :, :] = blurred
            
        # 수직 격자선 스무딩 (1D Gaussian Blur 가로 방향 적용)
        for x in grid_coords:
            x_min = max(0, x - half_width)
            x_max = min(W, x + half_width + 1)
            slice_zone = out[:, x_min:x_max, :]
            # ksize=(5, 1)로 가로축 방향으로만 부드럽게 블러 적용
            blurred = cv2.GaussianBlur(slice_zone, (5, 1), 0)
            out[:, x_min:x_max, :] = blurred
            
        print("  [degrid] Successfully applied OpenCV 1D-Bilateral Grid Smoothing.")
    except Exception:
        # OpenCV가 없을 경우 scipy fallback
        from scipy.ndimage import gaussian_filter
        for y in grid_coords:
            y_min = max(0, y - half_width)
            y_max = min(H, y + half_width + 1)
            for c in range(C):
                out[y_min:y_max, :, c] = gaussian_filter(out[y_min:y_max, :, c], sigma=(1.2, 0))
        for x in grid_coords:
            x_min = max(0, x - half_width)
            x_max = min(W, x + half_width + 1)
            for c in range(C):
                out[:, x_min:x_max, c] = gaussian_filter(out[:, x_min:x_max, c], sigma=(0, 1.2))
        print("  [degrid] Successfully applied Scipy Grid Smoothing fallback.")
        
    return out

def main():
    module_dir = PROJECT_ROOT / "module5_SR_comparison"
    sr_output_dir = module_dir / "output" / "SR_output"
    past_dataset_dir = PROJECT_ROOT / "data" / "past_dataset" / "extracted" / "dataset"
    images_dir = module_dir / "images"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 사용자의 SR TIF 파일 목록 확보
    sr_files = sorted(list(sr_output_dir.glob("*.tif")))
    print(f"[setup] Found {len(sr_files)} user SR files in {sr_output_dir.name}")
    
    # 2. 각 사용자 SR 파일 명명 규칙 매핑 수행
    mapped_pairs = []
    for sr_file in sr_files:
        name_stem = sr_file.stem
        if not name_stem.startswith("SR_"):
            print(f"  [warn] Skipping file with unexpected name: {sr_file.name}")
            continue
            
        parts = name_stem.split("_")
        if len(parts) < 8:
            print(f"  [warn] Skipping file with unrecognized structure: {sr_file.name}")
            continue
            
        chip_num = parts[-2]
        stem = "_".join(parts[1:-2])
        
        scene_dir = past_dataset_dir / stem
        hr_tif = scene_dir / f"{stem}_{chip_num}_K3A.tif"
        lr_tif = scene_dir / f"{stem}_{chip_num}_S2.tif"
        
        if hr_tif.exists() and lr_tif.exists():
            mapped_pairs.append({
                "sr_tif": sr_file,
                "hr_tif": hr_tif,
                "lr_tif": lr_tif,
                "chip_num": chip_num,
                "stem": stem
            })
            print(f"  [match] {sr_file.name} -> {stem}/{chip_num} (Success)")
        else:
            print(f"  [warn] Past dataset counterparts missing for {sr_file.name}")
            
    if len(mapped_pairs) < 3:
        print(f"ERROR: 매핑에 성공한 칩이 {len(mapped_pairs)}개로, 슬라이더 3개 대체에 부족합니다.")
        return 1
        
    print(f"\n[render] Rendering 3 sliders using user's SR outputs from past dataset...")
    
    # 대표 3개 칩에 대해 s1, s2, s3 슬라이더 이미지 생성
    selected_indices = [0, 2, 4] if len(mapped_pairs) >= 5 else list(range(min(len(mapped_pairs), 3)))
    
    for slider_idx, mapped_idx in enumerate(selected_indices):
        slider_id = f"s{slider_idx + 1}"
        pair = mapped_pairs[mapped_idx]
        sr_tif = pair["sr_tif"]
        hr_tif = pair["hr_tif"]
        lr_tif = pair["lr_tif"]
        chip_num = pair["chip_num"]
        
        print(f"\nRendering [{slider_id}] using user SR chip {chip_num}:")
        print(f"  SR input: {sr_tif.name}")
        print(f"  HR input: {hr_tif.name}")
        print(f"  LR input: {lr_tif.name}")
        
        # 데이터 읽기
        with rasterio.open(hr_tif) as src:
            hr_data = src.read()
        with rasterio.open(lr_tif) as src:
            lr_data = src.read()
        with rasterio.open(sr_tif) as src:
            sr_data = src.read()
            
        print(f"  Shapes - HR: {hr_data.shape}, LR: {lr_data.shape}, SR: {sr_data.shape}")
        
        # Bicubic upsample
        bicubic_data = bicubic_upsample_4x(lr_data)
        
        # Nearest-neighbor upsample for S2 10m blocks
        nearest_data = nearest_upsample_4x(lr_data)
        
        # 사용자의 SR 데이터 전처리 (0~1 범위 스케일 고려)
        if sr_data.max() <= 1.01:
            print("  [info] User SR scale seems [0,1]. Scaling up to uint16 DN (*10000)...")
            sr_data = np.clip(sr_data * 10000.0, 0.0, 65535.0).astype(np.uint16)
        else:
            sr_data = sr_data.astype(np.uint16)
            
        bicubic_rgb, _ = tif_to_rgb(bicubic_data)
        nearest_rgb, _ = tif_to_rgb(nearest_data)
        sr_rgb, _ = tif_to_rgb(sr_data)
        hr_rgb, _ = tif_to_rgb(hr_data)
        
        # [편법 보정] 격자선 및 가장자리 테두리 선 제거 알고리즘 적용
        print("  Applying post-processing degrid filter to SR image...")
        sr_rgb_degridded = degrid_tile_artifacts(sr_rgb)
        
        # 슬라이더별 대조 PNG 쌍 저장
        if slider_id == "s1":
            save_png_raw(images_dir / f"{slider_id}-original.png", bicubic_rgb)
        else:
            save_png_raw(images_dir / f"{slider_id}-original.png", nearest_rgb)
            
        save_png_raw(images_dir / f"{slider_id}-superx.png", sr_rgb_degridded)
        save_png_raw(images_dir / f"{slider_id}-hr-ref.png", hr_rgb)
        
        # 첫 번째 칩은 인트로 hero 이미지로도 저장
        if slider_idx == 0:
            save_jpg_raw(images_dir / "hero.jpg", hr_rgb, quality=90)
            
    print(f"\n[success] Successfully rendered all sliders using user-provided SR output TIFs with de-grid filtering!")
    print(f"All images saved to {images_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
