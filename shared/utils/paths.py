"""paths.py - 프로젝트 루트 + config/paths.json 로딩 헬퍼 (모든 모듈 공용)

PROJECT_ROOT 는 이 파일 기준 상위 2단계 (shared/utils/paths.py → 프로젝트 루트)
로 고정. 모든 데이터/설정 경로 해석의 기준점.
"""

import json
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
"""프로젝트 루트 디렉토리 (config/, data/, module*_/, shared/ 의 부모)."""

_CONFIG_PATH = PROJECT_ROOT / "config" / "paths.json"


def load_paths() -> dict:
    """config/paths.json 에서 경로 설정을 dict 로 읽어 반환합니다."""
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(key: str) -> Path:
    """paths.json 의 key 에 대응하는 경로를 PROJECT_ROOT 기준 절대경로로 반환."""
    return (PROJECT_ROOT / load_paths()[key]).resolve()
