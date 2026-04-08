"""
TableNormalizer 的异步包装器
提供异步执行和进度回调功能
"""

import sys
import threading
import logging
from pathlib import Path
import asyncio
from typing import Dict, Any, Optional, Callable

# 动态导入 spreadsheet-normalizer
normalizer_path = Path(__file__).parent.parent.parent.parent / "spreadsheet-normalizer"
sys.path.insert(0, str(normalizer_path))

from src.pipeline import TableNormalizer

logger = logging.getLogger(__name__)

# Pipeline stages: (stage_key, progress_pct, display_message)
PIPELINE_STAGES = [
    ("initializing",          5,  "Initializing pipeline..."),
    ("encoding",             20,  "Stage 1 / 4 — Spreadsheet Encoding"),
    ("irregularity_detection", 45, "Stage 2 / 4 — Irregularity Detection"),
    ("schema_estimation",    65,  "Stage 3 / 4 — Schema Estimation"),
    ("transformation",       85,  "Stage 4 / 4 — Transformation Generation"),
    ("completed",           100,  "Normalization completed successfully."),
]


class NormalizerWrapper:
    """TableNormalizer 的异步包装器"""

    def __init__(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None):
        """
        Args:
            config: TableNormalizer 配置字典
            progress_callback: (stage: str, progress: int, message: str) -> None
        """
        self.config = config
        self.progress_callback = progress_callback
        self.normalizer = None

    def _report(self, stage: str, progress: int, message: str):
        if self.progress_callback:
            self.progress_callback(stage, progress, message)

    async def normalize_async(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._normalize_sync,
            input_file,
            output_file,
        )

    def _normalize_sync(self, input_file: str, output_file: Optional[str]) -> Dict[str, Any]:
        try:
            self._report("initializing", 5, "Initializing pipeline...")
            self.normalizer = TableNormalizer(self.config)
            self._report("encoding", 20, "Stage 1 / 4 — Spreadsheet Encoding")

            # Inject progress hooks into the pipeline via log interception
            _inject_progress_hooks(self.normalizer, self._report)

            result = self.normalizer.normalize(input_file, output_file)

            self._report("completed", 100, "Normalization completed successfully.")
            return result

        except Exception as e:
            logger.error(f"Normalization failed: {e}", exc_info=True)
            self._report("failed", 0, f"Error: {str(e)}")
            raise


def _inject_progress_hooks(normalizer: TableNormalizer, report: Callable):
    """
    Monkey-patch the pipeline's private stage methods to emit progress
    callbacks before each stage runs.
    """
    original_detection  = normalizer._run_detection
    original_schema     = normalizer._run_schema_estimation
    original_transform  = normalizer._run_transformation

    def hooked_detection(*args, **kwargs):
        report("irregularity_detection", 45, "Stage 2 / 4 — Irregularity Detection")
        return original_detection(*args, **kwargs)

    def hooked_schema(*args, **kwargs):
        report("schema_estimation", 65, "Stage 3 / 4 — Schema Estimation")
        return original_schema(*args, **kwargs)

    def hooked_transform(*args, **kwargs):
        report("transformation", 85, "Stage 4 / 4 — Transformation Generation")
        return original_transform(*args, **kwargs)

    normalizer._run_detection         = hooked_detection
    normalizer._run_schema_estimation = hooked_schema
    normalizer._run_transformation    = hooked_transform
