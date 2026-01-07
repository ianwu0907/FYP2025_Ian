"""
DataFrame 转换工具
将 pandas DataFrame 转换为 JSON-safe 格式
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


class DataFrameConverter:
    """DataFrame 转换器"""

    @staticmethod
    def to_json_preview(df: pd.DataFrame, max_rows: int = 100) -> Dict[str, Any]:
        """
        转换 DataFrame 为 JSON 预览格式

        Args:
            df: pandas DataFrame
            max_rows: 最大预览行数

        Returns:
            JSON-safe 字典
        """
        # 处理 NaN 值
        preview_df = df.head(max_rows).replace({np.nan: None})

        return {
            "shape": list(df.shape),
            "columns": df.columns.tolist(),
            "data": preview_df.to_dict(orient='records'),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "null_counts": df.isnull().sum().to_dict(),
            "total_rows": len(df),
            "preview_rows": len(preview_df)
        }

    @staticmethod
    def prepare_comparison(original_df: pd.DataFrame, normalized_df: pd.DataFrame) -> Dict[str, Any]:
        """
        准备原始 vs 标准化对比数据

        Args:
            original_df: 原始 DataFrame
            normalized_df: 标准化后的 DataFrame

        Returns:
            对比数据字典
        """
        return {
            "original": DataFrameConverter.to_json_preview(original_df),
            "normalized": DataFrameConverter.to_json_preview(normalized_df),
            "changes": {
                "shape_changed": original_df.shape != normalized_df.shape,
                "columns_changed": list(original_df.columns) != list(normalized_df.columns),
                "original_columns": original_df.columns.tolist(),
                "normalized_columns": normalized_df.columns.tolist(),
                "row_count_diff": len(normalized_df) - len(original_df),
                "column_count_diff": len(normalized_df.columns) - len(original_df.columns)
            }
        }

    @staticmethod
    def dataframe_to_csv_string(df: pd.DataFrame) -> str:
        """
        将 DataFrame 转换为 CSV 字符串

        Args:
            df: pandas DataFrame

        Returns:
            CSV 格式字符串
        """
        return df.to_csv(index=False)

    @staticmethod
    def safe_serialize(obj: Any) -> Any:
        """
        安全序列化对象（处理 numpy 类型等）

        Args:
            obj: 待序列化对象

        Returns:
            JSON-safe 对象
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        else:
            return obj
