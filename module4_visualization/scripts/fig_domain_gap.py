"""fig_domain_gap.py - 도메인 갭 4-패널 가로 figure.

[K3A 원본 2.5m] | [단순 보간 10m] | [영상모사 10m] | [Sentinel-2 원본 10m]

목적: K3A 원본을 단순히 다운샘플한 영상 vs 방사+MTF 모사를 거친 영상 vs S2 원본 사이의
"도메인 갭" 을 시각적으로 확인.

스트레치 정책 (기본 = group-shared):
  - Group A: (a) K3A 원본 + (b) 단순 보간 — 두 패널이 K3A raw DN 공유, 동일 채널별 vmin/vmax
  - Group B: (c) 영상모사 + (d) S2 원본 — 두 패널이 S2 reflectance 공유, 동일 채널별 vmin/vmax
  -> 모사가 잘 됐다면 (c) 와 (d) 의 톤이 시각적으로 일치해야 한다.

사용:
  python module4_visualization/scripts/fig_domain_gap.py --label 10
  python module4_visualization/scripts/fig_domain_gap.py --label 10 --center-crop 0.5
  python module4_visualization/scripts/fig_domain_gap.py --label 10 --bbox <minx> <miny> <maxx> <maxy>
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from module4_visualization.src import (
    aggregate_4x4_mean,
    bilinear_4x_downsample,
    locate_pair,
    percentile_stretch,
    read_rgb_stack,
    read_simulated_rgb,
)
from shared.utils.paths import resolve_path


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_bbox(values: Optional[list]) -> Optional[Tuple[float, float, float, float]]:
    if values is None:
        return None
    if len(values) != 4:
        raise ValueError("--bbox 는 4개 값 필요: minx miny maxx maxy")
    return tuple(float(v) for v in values)  # type: ignore[return-value]


def center_crop(stack: np.ndarray, ratio: float) -> np.ndarray:
    """(3,H,W) 의 중앙을 ratio 비율로 crop (가로/세로 동일 비율)."""
    if ratio <= 0.0 or ratio >= 1.0:
        return stack
    _, h, w = stack.shape
    new_h = max(1, int(round(h * ratio)))
    new_w = max(1, int(round(w * ratio)))
    y0 = (h - new_h) // 2
    x0 = (w - new_w) // 2
    return stack[:, y0:y0 + new_h, x0:x0 + new_w]


def _per_channel_vmin_vmax(stacks: List[np.ndarray], pmin: float, pmax: float, nodata_value: float = 0.0) -> np.ndarray:
    """여러 (3,H,W) 스택을 채널별로 합쳐 (3,2) [[vmin,vmax]_R,G,B] percentile 반환.

    그룹 내 모든 패널이 동일 vmin/vmax 를 쓰게 만드는 게 목적.
    """
    rng = np.random.default_rng(0)
    out = np.zeros((3, 2), dtype=np.float32)
    for c in range(3):
        chunks = []
        for s in stacks:
            ch = s[c].astype(np.float32).ravel()
            valid = ch[ch != nodata_value]
            if valid.size == 0:
                continue
            sample = valid if valid.size <= 2_000_000 else rng.choice(valid, 2_000_000, replace=False)
            chunks.append(sample)
        if not chunks:
            out[c] = (0.0, 1.0)
            continue
        cat = np.concatenate(chunks)
        vmin = float(np.percentile(cat, pmin))
        vmax = float(np.percentile(cat, pmax))
        if vmax <= vmin:
            vmax = vmin + 1.0
        out[c] = (vmin, vmax)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="도메인 갭 4-패널 figure")
    parser.add_argument("--label", type=int, required=True, help="페어 label (예: 10)")
    parser.add_argument("--bbox", nargs=4, default=None, help="UTM bbox: minx miny maxx maxy (기본: 전체)")
    parser.add_argument("--center-crop", type=float, default=0.5, help="UTM bbox 후 중앙 crop 비율 (0~1, 기본 0.5). 0/1 이면 crop 안함")
    parser.add_argument("--out", type=Path, default=None, help="출력 PNG 경로 (기본: data/output/figures/domain_gap_label{N}.png)")
    parser.add_argument("--pmin", type=float, default=2.0, help="퍼센타일 최소 (기본 2)")
    parser.add_argument("--pmax", type=float, default=98.0, help="퍼센타일 최대 (기본 98)")
    parser.add_argument(
        "--stretch-mode",
        choices=["group", "all-shared", "per-panel"],
        default="group",
        help="group=(a)(b)|(c)(d) 두 그룹 각자 공유 (기본), all-shared=4패널 전부 공유, per-panel=패널별 독립",
    )
    parser.add_argument("--simple-method", choices=["bilinear", "mean"], default="bilinear", help="단순 보간 방식")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--figwidth", type=float, default=16.0, help="figure 가로 inch (기본 16)")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("fig_domain_gap")

    bbox = parse_bbox(args.bbox)
    pair = locate_pair(args.label)
    if not pair.has_simulated():
        log.error("label=%d 에 simulated_final_2p5m.tif 가 없습니다: %s", args.label, pair.simulated)
        sys.exit(1)

    log.info("[label=%d] 데이터 로드 시작 (bbox=%s, center_crop=%s)", args.label, bbox, args.center_crop)
    k3a_25, _ = read_rgb_stack(pair.k3a_grid, ("R", "G", "B"), bbox)
    if args.simple_method == "bilinear":
        simple_10 = bilinear_4x_downsample(k3a_25, nodata_value=0.0)
    else:
        simple_10 = aggregate_4x4_mean(k3a_25)
    sim_25, _ = read_simulated_rgb(pair.simulated, bbox)
    sim_10 = aggregate_4x4_mean(sim_25)
    s2_10, _ = read_rgb_stack(pair.s2_grid, ("B04", "B03", "B02"), bbox)

    # 그룹 스트레치 (crop 이전 전체 영역 기준)
    if args.stretch_mode == "group":
        vv_a = _per_channel_vmin_vmax([k3a_25, simple_10], args.pmin, args.pmax)
        vv_b = _per_channel_vmin_vmax([sim_10, s2_10], args.pmin, args.pmax)
        log.info("  group A (K3A raw) vmin/vmax per ch: %s", vv_a.tolist())
        log.info("  group B (S2 ref ) vmin/vmax per ch: %s", vv_b.tolist())
        panel_vv = [vv_a, vv_a, vv_b, vv_b]
    elif args.stretch_mode == "all-shared":
        vv_all = _per_channel_vmin_vmax([k3a_25, simple_10, sim_10, s2_10], args.pmin, args.pmax)
        log.info("  all-shared vmin/vmax per ch: %s", vv_all.tolist())
        panel_vv = [vv_all] * 4
    else:  # per-panel
        panel_vv = [
            _per_channel_vmin_vmax([k3a_25], args.pmin, args.pmax),
            _per_channel_vmin_vmax([simple_10], args.pmin, args.pmax),
            _per_channel_vmin_vmax([sim_10], args.pmin, args.pmax),
            _per_channel_vmin_vmax([s2_10], args.pmin, args.pmax),
        ]

    # 중앙 crop (그룹 동일 ratio) — 스트레치 후 적용
    k3a_25 = center_crop(k3a_25, args.center_crop)
    simple_10 = center_crop(simple_10, args.center_crop)
    sim_10 = center_crop(sim_10, args.center_crop)
    s2_10 = center_crop(s2_10, args.center_crop)
    log.info("  shapes — K3A:%s simple:%s sim:%s S2:%s", k3a_25.shape, simple_10.shape, sim_10.shape, s2_10.shape)

    panels = [
        ("(a) Kompsat-3A original\n2.5 m", k3a_25),
        (f"(b) Simple {args.simple_method}\n2.5 -> 10 m", simple_10),
        ("(c) Image simulation\n(rad + MTF), 10 m", sim_10),
        ("(d) Sentinel-2 original\n10 m", s2_10),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(args.figwidth, args.figwidth / 4.0 * 1.1))
    for ax, (title, stack), vv in zip(axes, panels, panel_vv):
        rgb01, _ = percentile_stretch(stack, args.pmin, args.pmax, vmin_vmax=vv, per_channel=True)
        ax.imshow(rgb01, interpolation="nearest")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Domain gap visualization — pair label {args.label}  (stretch={args.stretch_mode})", fontsize=12, y=1.02)
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        out_dir = resolve_path("output_dir") / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"domain_gap_label{args.label}.png"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("저장: %s", out_path)


if __name__ == "__main__":
    main()
