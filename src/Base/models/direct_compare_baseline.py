from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PTL_CORE_PATH = REPO_ROOT / "src" / "PTL" / "progressive_core.py"


def _load_ptl_core():
    spec = importlib.util.spec_from_file_location("ptl_progressive_core_for_base", PTL_CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise ImportError(f"无法加载 PTL progressive_core: {PTL_CORE_PATH}")
    spec.loader.exec_module(module)
    return module


PTL_CORE = _load_ptl_core()


class DirectDailyTransformerBaseline(PTL_CORE.WaterQualityTransformer):
    """
    A fair comparison baseline for PTL:
    - same single-station backbone family as PTL stage model
    - no progressive stages
    - no sampler / postprocess / task-specific auxiliary heads by default
    """

    def __init__(
        self,
        num_heads: int,
        e_layer: int,
        hidden_size: int,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        target_dim: int | None = None,
        target_feature_names: list[str] | None = None,
        use_temporal_adapter: bool = True,
        temporal_adapter_kernel_size: int = 5,
    ):
        super().__init__(
            num_heads=num_heads,
            e_layer=e_layer,
            hidden_size=hidden_size,
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            target_dim=target_dim,
            target_feature_names=target_feature_names,
            use_temporal_adapter=use_temporal_adapter,
            temporal_adapter_kernel_size=temporal_adapter_kernel_size,
            nh4n_two_stage_config=None,
        )
