"""
Pydantic 请求模型
定义 API 请求的数据结构
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


class NormalizeRequest(BaseModel):
    """标准化请求模型"""

    session_id: str = Field(..., description="Session ID from upload")
    config_overrides: Optional[Dict[str, Any]] = Field(
        default={},
        description="Configuration overrides for the normalizer"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "config_overrides": {
                    "llm": {
                        "llm_provider": "gemini",
                        "model": "gemini-2.0-flash"
                    },
                    "table_splitting": {
                        "enabled": True,
                        "detection_mode": "auto"
                    }
                }
            }
        }


class ConfigUpdateRequest(BaseModel):
    """配置更新请求模型"""

    llm: Optional[Dict[str, Any]] = Field(default=None, description="LLM configuration")
    table_splitting: Optional[Dict[str, Any]] = Field(default=None, description="Table splitting configuration")
    analyzer: Optional[Dict[str, Any]] = Field(default=None, description="Analyzer configuration")
    estimator: Optional[Dict[str, Any]] = Field(default=None, description="Estimator configuration")
    generator: Optional[Dict[str, Any]] = Field(default=None, description="Generator configuration")
    output: Optional[Dict[str, Any]] = Field(default=None, description="Output configuration")

    class Config:
        json_schema_extra = {
            "example": {
                "llm": {
                    "llm_provider": "openai",
                    "model": "gpt-4"
                }
            }
        }
