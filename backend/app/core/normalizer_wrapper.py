"""
TableNormalizer 的异步包装器
提供异步执行和进度回调功能
"""

import sys
from pathlib import Path
import asyncio
import logging
from typing import Dict, Any, Optional, Callable

# 动态导入 spreadsheet-normalizer
normalizer_path = Path(__file__).parent.parent.parent.parent / "spreadsheet-normalizer"
sys.path.insert(0, str(normalizer_path))

from src.pipeline import TableNormalizer

logger = logging.getLogger(__name__)


class NormalizerWrapper:
    """TableNormalizer 的异步包装器"""

    def __init__(self, config: Dict[str, Any], progress_callback: Optional[Callable] = None):
        """
        初始化包装器

        Args:
            config: TableNormalizer 配置字典
            progress_callback: 进度回调函数 (stage: str, progress: int, message: str) -> None
        """
        self.config = config
        self.progress_callback = progress_callback
        self.normalizer = None

    async def normalize_async(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        异步执行标准化

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）

        Returns:
            标准化结果字典
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._normalize_sync,
            input_file,
            output_file
        )

    def _normalize_sync(self, input_file: str, output_file: Optional[str]) -> Dict[str, Any]:
        """
        同步执行标准化（在线程池中运行）

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选）

        Returns:
            标准化结果字典
        """
        try:
            # Stage 0: Initializing (0-10%)
            if self.progress_callback:
                self.progress_callback("initializing", 5, "Initializing normalizer pipeline...")

            # 初始化 TableNormalizer，传递进度回调
            # TableNormalizer 会在每个真实阶段完成时调用这个回调
            self.normalizer = TableNormalizer(self.config, progress_callback=self.progress_callback)

            # 执行标准化（TableNormalizer 内部会报告 10-100% 的进度）
            result = self.normalizer.normalize(input_file, output_file)

            return result

        except Exception as e:
            logger.error(f"Normalization failed: {e}", exc_info=True)
            if self.progress_callback:
                self.progress_callback("failed", 0, f"Error: {str(e)}")
            raise

    def get_progress_info(self) -> Dict[str, Any]:
        """
        获取当前进度信息

        Returns:
            进度信息字典
        """
        # TODO: 实现更详细的进度跟踪
        return {
            "status": "unknown",
            "progress": 0,
            "message": "No active task"
        }
