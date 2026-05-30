import os
import sys

# DLL loading path and OpenMP duplication workarounds for Windows
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

# MUST import torch before other spatial modules
try:
    import torch
except Exception:
    pass

"""module5_SR_comparison.src - SR downstream comparison library.

공개 API:
  - bicubic_sr, prithvi_sr : 각 method 별 4× SR 함수 (numpy in / out)
  - FullSRModel, get_prithvi_encoder_with_lora, load_prithvi_sr_model : Prithvi-SR 모델
  - tile_stitch_inference : 큰 입력을 fixed-size 모델로 처리
"""

from module5_SR_comparison.src.sr_inference import (  # noqa: F401
    bicubic_sr,
    prithvi_sr,
)
from module5_SR_comparison.src.tile_stitch import tile_stitch_inference  # noqa: F401
